// src/time.rs

use std::time::{Duration, Instant};
use cozy_chess::Move;

use crate::search::TTNodeType;

#[derive(Clone, Debug, PartialEq)]
pub struct TimeConstants {
    pub move_overhead_ms: u64,
    pub min_time_ms: u64,
    pub max_time_ratio: f64,
    pub sudden_death_opt_scale: f64,
    pub max_to_opt_ratio: f64,
    pub increment_scale: f64,
    pub max_total_extension: f64,
    pub min_total_reduction: f64,
    pub fail_low_penalty: f64,
    pub max_fail_low_factor: f64,
    pub score_falling_threshold: i32,
    pub score_falling_factor: f64,
}

impl Default for TimeConstants {
    fn default() -> Self {
        Self {
            move_overhead_ms: 30,
            min_time_ms: 1,
            max_time_ratio: 0.8,
            sudden_death_opt_scale: 0.003,
            max_to_opt_ratio: 4.6,
            increment_scale: 0.9,
            max_total_extension: 1.7,
            min_total_reduction: 0.65,
            fail_low_penalty: 1.0,
            max_fail_low_factor: 1.65,
            score_falling_threshold: -19,
            score_falling_factor: 1.1,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ForcedMoveType {
    OneLegal,
    None,
}

#[derive(PartialEq, Clone, Debug)]
pub enum SearchLimit {
    Infinite,
    Depth(usize),
    Time(u64), 
    Nodes(u64),
    Dynamic {
        our_clock: u64,
        our_inc: u64,
        moves_to_go: Option<u64>,
        constants: TimeConstants,
    },
}

impl Default for SearchLimit {
    fn default() -> Self {
        Self::Infinite
    }
}

impl SearchLimit {
    pub fn from_uci(time_left: Option<Duration>, inc: Option<Duration>, moves_to_go: Option<u32>, movetime: Option<Duration>, constants: TimeConstants) -> Self {
        if let Some(mt) = movetime {
            return Self::Time(mt.as_millis() as u64);
        }
        if let Some(t) = time_left {
            return Self::Dynamic {
                our_clock: t.as_millis() as u64,
                our_inc: inc.map(|i| i.as_millis() as u64).unwrap_or(0),
                moves_to_go: moves_to_go.map(|m| m as u64),
                constants,
            };
        }
        Self::Infinite
    }
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    pub start_time: Instant,
    pub limit: SearchLimit,
    pub constants: TimeConstants,
    
    pub base_opt_time: Duration,
    pub current_opt_time: Duration,
    pub max_time: Duration,
    pub allow_scaling: bool,
    
    pub prev_score: i32,
    pub prev_move: Option<Move>,
    pub stability: usize,
    pub fail_low_streak: u32,
    pub score_drop_streak: u32,
    pub found_forced_move: ForcedMoveType,
    pub best_move_nodes_fraction: Option<f64>,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            limit: SearchLimit::Infinite,
            constants: TimeConstants::default(),
            base_opt_time: Duration::ZERO,
            current_opt_time: Duration::ZERO,
            max_time: Duration::MAX,
            allow_scaling: false,
            prev_score: 0,
            prev_move: None,
            stability: 0,
            fail_low_streak: 0,
            score_drop_streak: 0,
            found_forced_move: ForcedMoveType::None,
            best_move_nodes_fraction: None,
        }
    }
}

impl TimeManager {
    pub fn new(limit: SearchLimit) -> Self {
        let constants = match &limit {
            SearchLimit::Dynamic { constants: c, .. } => c.clone(),
            _ => TimeConstants::default(),
        };

        let mut tm = Self {
            limit: limit.clone(),
            constants,
            ..Default::default()
        };
        tm.init_limits();
        tm
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
    }

    fn init_limits(&mut self) {
        match &self.limit {
            SearchLimit::Time(ms) => {
                let safe_ms = ms.saturating_sub(self.constants.move_overhead_ms).max(self.constants.min_time_ms);
                let dur = Duration::from_millis(safe_ms);
                self.base_opt_time = dur;
                self.current_opt_time = dur;
                self.max_time = dur;
                self.allow_scaling = false;
            }
            SearchLimit::Dynamic { our_clock, our_inc, moves_to_go, constants } => {
                let t = *our_clock as f64;
                let inc = *our_inc as f64;
                let oh = constants.move_overhead_ms as f64;
                let safe_t = (t - oh).max(constants.min_time_ms as f64);

                let (opt_ms, max_ms) = match moves_to_go {
                    Some(mtg) => {
                        let mtg = (*mtg).max(1) as f64;
                        let total = safe_t + inc * (mtg - 1.0).max(0.0);
                        let opt = (total / (mtg + 1.0)).min(safe_t * 0.9);
                        let max = (opt * constants.max_to_opt_ratio).min(safe_t * constants.max_time_ratio);
                        (opt, max)
                    }
                    None => {
                        let moves_left = 30.0; 
                        let opt = (safe_t * constants.sudden_death_opt_scale + inc * constants.increment_scale)
                            .min(safe_t / moves_left * 2.5)
                            .max(constants.min_time_ms as f64);
                        let max = (opt * constants.max_to_opt_ratio).min(safe_t * constants.max_time_ratio);
                        (opt, max)
                    }
                };

                self.base_opt_time = Duration::from_millis(opt_ms as u64);
                self.current_opt_time = self.base_opt_time;
                self.max_time = Duration::from_millis(max_ms as u64);
                self.allow_scaling = true; 
            }
            _ => {
                self.base_opt_time = Duration::MAX;
                self.current_opt_time = Duration::MAX;
                self.max_time = Duration::MAX;
                self.allow_scaling = false;
            }
        }
    }

    pub fn is_past_opt_time(&self) -> bool {
        if self.found_forced_move == ForcedMoveType::OneLegal {
            return true; 
        }
        match self.limit {
            SearchLimit::Dynamic { .. } | SearchLimit::Time(_) => {
                self.start_time.elapsed() >= self.current_opt_time
            }
            _ => false,
        }
    }

    pub fn is_hard_stop(&self) -> bool {
        match self.limit {
            SearchLimit::Dynamic { .. } | SearchLimit::Time(_) => {
                self.start_time.elapsed() >= self.max_time
            }
            _ => false,
        }
    }

    pub fn notify_one_legal_move(&mut self) {
        self.found_forced_move = ForcedMoveType::OneLegal;
        self.current_opt_time = Duration::ZERO;
    }

    pub fn report_completed_depth(
        &mut self,
        _depth: i32,
        eval: i32,
        best_move: Move,
        best_move_nodes_fraction: Option<f64>,
    ) {
        if !self.allow_scaling { return; }

        if Some(best_move) == self.prev_move {
            self.stability += 1;
        } else {
            self.stability = 0;
        }

        if self.prev_move.is_some() && eval - self.prev_score < self.constants.score_falling_threshold {
            self.score_drop_streak += 1;
        } else if eval - self.prev_score > 15 {
            self.score_drop_streak = 0;
        }

        self.prev_move = Some(best_move);
        self.prev_score = eval;
        self.best_move_nodes_fraction = best_move_nodes_fraction;

        self.recompute_optimum();
    }

    pub fn report_aspiration_fail(&mut self, depth: i32, bound: TTNodeType) {
        if !self.allow_scaling { return; }
        
        const FAIL_LOW_UPDATE_THRESHOLD: i32 = 4;
        if depth >= FAIL_LOW_UPDATE_THRESHOLD && bound == TTNodeType::Alpha {
            self.fail_low_streak += 1;
            self.recompute_optimum();
        }
    }

    fn recompute_optimum(&mut self) {
        if !self.allow_scaling { return; }

        let mut multiplier = 1.0;

        multiplier *= match self.stability {
            0 => 1.25,
            1 => 1.05,
            2 => 0.95,
            3 => 0.85,
            _ => 0.80,
        };

        if let Some(frac) = self.best_move_nodes_fraction {
            if frac > 0.60 {
                multiplier *= 0.85; 
            } else if frac < 0.30 {
                multiplier *= 1.15; 
            }
        }

        if self.fail_low_streak > 0 {
            let fail_factor = self.constants.fail_low_penalty.powi(self.fail_low_streak as i32);
            multiplier *= fail_factor.min(self.constants.max_fail_low_factor);
        }
        if self.score_drop_streak > 0 {
            multiplier *= self.constants.score_falling_factor;
        }

        multiplier = multiplier.clamp(self.constants.min_total_reduction, self.constants.max_total_extension);

        let new_opt_ms = (self.base_opt_time.as_millis() as f64 * multiplier) as u64;
        self.current_opt_time = Duration::from_millis(new_opt_ms).min(self.max_time);
    }
}
