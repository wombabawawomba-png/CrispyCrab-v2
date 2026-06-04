// src/uci.rs

use cozy_chess::{Board, GameStatus, Move, Piece, Square};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use vampirc_uci::{parse_with_unknown, UciMessage};

use crate::search::{find_best_move, TranspositionTable, SearchConstants, MAX_SEARCH_DEPTH, GLOBAL_HISTORY};
use crate::time::{SearchLimit, TimeConstants};

pub fn run_uci() {
    let tt = Arc::new(TranspositionTable::new(64));
    let mut board = Board::default();
    let mut history_hashes = Vec::new();
    let mut stop_flag = Arc::new(AtomicBool::new(false));
    let is_pondering = Arc::new(AtomicBool::new(false));
    let mut search_thread: Option<std::thread::JoinHandle<()>> = None;
    
    let mut time_constants = TimeConstants::default();
    let mut search_constants = SearchConstants::default();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let messages = parse_with_unknown(&line);

        for msg in messages {
            match msg {
                UciMessage::Uci => {
                    println!("id name CrispyCrab");
                    println!("id author You");
                    
                    println!("option name Ponder type check default true");
                    
                    // Time Options
                    println!("option name MaxTimeRatio type string default 0.85");
                    println!("option name SuddenDeathOptScale type string default 0.033");
                    println!("option name MaxToOptRatio type string default 5.0");
                    println!("option name IncrementScale type string default 0.80");
                    println!("option name MaxTotalExtension type string default 1.80");
                    println!("option name MinTotalReduction type string default 0.70");
                    println!("option name FailLowPenalty type string default 1.15");
                    println!("option name MaxFailLowFactor type string default 1.40");
                    println!("option name ScoreFallingThreshold type string default -25");
                    println!("option name ScoreFallingFactor type string default 1.15");
                    
                    // Search Options
                    println!("option name AspirationDelta type string default 25");
                    println!("option name MaxDelta type string default 300");
                    println!("option name MaxDeltaCapture type string default 1150");
                    println!("option name QsFutilityMargin type string default 150");
                    println!("option name SeeThreshold type string default -100");
                    println!("option name RazorMargin type string default 300");
                    println!("option name ProbcutMargin type string default 200");
                    println!("option name SingularDepthThreshold type string default 8");
                    println!("option name SingularMarginMultiplier type string default 2");
                    println!("option name TempoBonus type string default 15");
                    println!("option name RfpMultiplier type string default 75");
                    println!("option name FutilityMultiplier type string default 150");
                    println!("option name SeeQuietMultiplier type string default -200");
                    println!("option name HistoryPruningMultiplier type string default -4000");
                    println!("option name LmrBase type string default 0.75");
                    println!("option name LmrDivisor type string default 2.25");

                    // SPSA Tunable NMP Options (Exposing Divisor-based logic)
                    println!("option name NmpBase type string default 3");
                    println!("option name NmpDepthDiv type string default 4");
                    println!("option name NmpEvalDiv type string default 200");
                    println!("option name NmpEvalLimit type string default 3");

                    // --- Dynamic Generation of SPSA Tunable PST Options (96 parameters total) ---
                    for i in 0..32 {
                        println!("option name pst_knight_{} type string default {}", i, crate::weigh::TUNABLE_KNIGHT_PST[i].load(Ordering::Relaxed));
                        println!("option name pst_pawnmg_{} type string default {}", i, crate::weigh::TUNABLE_PAWN_MG[i].load(Ordering::Relaxed));
                        println!("option name pst_pawneg_{} type string default {}", i, crate::weigh::TUNABLE_PAWN_EG[i].load(Ordering::Relaxed));
                    }
                    
                    println!("uciok");
                }
                UciMessage::IsReady => {
                    println!("readyok");
                }
                UciMessage::SetOption { name, value } => {
                    let val_str = value.unwrap_or_default();
                    let name_lower = name.to_lowercase();

                    // Dynamic parsing of SPSA Tunable PST parameters first
                    if name_lower.starts_with("pst_knight_") {
                        if let Some(idx_str) = name_lower.strip_prefix("pst_knight_") {
                            if let Ok(idx) = idx_str.parse::<usize>() {
                                if idx < 32 {
                                    if let Ok(v) = val_str.parse::<i32>() {
                                        crate::weigh::TUNABLE_KNIGHT_PST[idx].store(v, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    } else if name_lower.starts_with("pst_pawnmg_") {
                        if let Some(idx_str) = name_lower.strip_prefix("pst_pawnmg_") {
                            if let Ok(idx) = idx_str.parse::<usize>() {
                                if idx < 32 {
                                    if let Ok(v) = val_str.parse::<i32>() {
                                        crate::weigh::TUNABLE_PAWN_MG[idx].store(v, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    } else if name_lower.starts_with("pst_pawneg_") {
                        if let Some(idx_str) = name_lower.strip_prefix("pst_pawneg_") {
                            if let Ok(idx) = idx_str.parse::<usize>() {
                                if idx < 32 {
                                    if let Ok(v) = val_str.parse::<i32>() {
                                        crate::weigh::TUNABLE_PAWN_EG[idx].store(v, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    } else {
                        // Standard matching for other UCI options
                        match name_lower.as_str() {
                            // Time
                            "maxtimeratio" => if let Ok(v) = val_str.parse() { time_constants.max_time_ratio = v; },
                            "suddendeathoptscale" => if let Ok(v) = val_str.parse() { time_constants.sudden_death_opt_scale = v; },
                            "maxtooptratio" => if let Ok(v) = val_str.parse() { time_constants.max_to_opt_ratio = v; },
                            "incrementscale" => if let Ok(v) = val_str.parse() { time_constants.increment_scale = v; },
                            "maxtotalextension" => if let Ok(v) = val_str.parse() { time_constants.max_total_extension = v; },
                            "mintotalreduction" => if let Ok(v) = val_str.parse() { time_constants.min_total_reduction = v; },
                            "faillowpenalty" => if let Ok(v) = val_str.parse() { time_constants.fail_low_penalty = v; },
                            "maxfaillowfactor" => if let Ok(v) = val_str.parse() { time_constants.max_fail_low_factor = v; },
                            "scorefallingthreshold" => if let Ok(v) = val_str.parse() { time_constants.score_falling_threshold = v; },
                            "scorefallingfactor" => if let Ok(v) = val_str.parse() { time_constants.score_falling_factor = v; },
                            
                            // Search
                            "aspirationdelta" => if let Ok(v) = val_str.parse() { search_constants.aspiration_delta = v; },
                            "maxdelta" => if let Ok(v) = val_str.parse() { search_constants.max_delta = v; },
                            "maxdeltacapture" => if let Ok(v) = val_str.parse() { search_constants.max_delta_capture = v; },
                            "qsfutilitymargin" => if let Ok(v) = val_str.parse() { search_constants.qs_futility_margin = v; },
                            "seethreshold" => if let Ok(v) = val_str.parse() { search_constants.see_threshold = v; },
                            "razormargin" => if let Ok(v) = val_str.parse() { search_constants.razor_margin = v; },
                            "probcutmargin" => if let Ok(v) = val_str.parse() { search_constants.probcut_margin = v; },
                            "singulardepththreshold" => if let Ok(v) = val_str.parse() { search_constants.singular_depth_threshold = v; },
                            "singularmarginmultiplier" => if let Ok(v) = val_str.parse() { search_constants.singular_margin_multiplier = v; },
                            "tempobonus" => if let Ok(v) = val_str.parse() { search_constants.tempo_bonus = v; },
                            "rfpmultiplier" => if let Ok(v) = val_str.parse() { search_constants.rfp_multiplier = v; },
                            "futilitymultiplier" => if let Ok(v) = val_str.parse() { search_constants.futility_multiplier = v; },
                            "seequietmultiplier" => if let Ok(v) = val_str.parse() { search_constants.see_quiet_multiplier = v; },
                            "historypruningmultiplier" => if let Ok(v) = val_str.parse() { search_constants.history_pruning_multiplier = v; },
                            "lmrbase" => if let Ok(v) = val_str.parse() { search_constants.lmr_base = v; },
                            "lmrdivisor" => if let Ok(v) = val_str.parse() { search_constants.lmr_divisor = v; },

                            // NMP Tunable Parameters
                            "nmpbase" => if let Ok(v) = val_str.parse() { search_constants.nmp_base = v; },
                            "nmpdepthdiv" | "nmpdepthcoef" => if let Ok(v) = val_str.parse() { search_constants.nmp_depth_div = v; },
                            "nmpevaldiv" | "nmpevalcoef" => if let Ok(v) = val_str.parse() { search_constants.nmp_eval_div = v; },
                            "nmpevallimit" => if let Ok(v) = val_str.parse() { search_constants.nmp_eval_limit = v; },
                            _ => {}
                        }
                    }
                }
                UciMessage::UciNewGame => {
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Some(handle) = search_thread.take() {
                        let _ = handle.join();
                    }
                    tt.clear();
                    GLOBAL_HISTORY.clear();
                    board = Board::default();
                    history_hashes.clear();
                }
                UciMessage::Position { startpos, fen, moves } => {
                    if startpos {
                        board = Board::default();
                    } else if let Some(f) = fen {
                        if let Ok(b) = Board::from_fen(&f.to_string(), false) {
                            board = b;
                        }
                    }
                    history_hashes.clear();
                    history_hashes.push(board.hash());

                    for m in moves {
                        let mut move_str = m.to_string();
                        
                        if move_str == "e1g1" && board.piece_on(Square::E1) == Some(Piece::King) {
                            move_str = "e1h1".to_string();
                        } else if move_str == "e1c1" && board.piece_on(Square::E1) == Some(Piece::King) {
                            move_str = "e1a1".to_string();
                        } else if move_str == "e8g8" && board.piece_on(Square::E8) == Some(Piece::King) {
                            move_str = "e8h8".to_string();
                        } else if move_str == "e8c8" && board.piece_on(Square::E8) == Some(Piece::King) {
                            move_str = "e8a8".to_string();
                        }

                        if let Ok(cozy_move) = Move::from_str(&move_str) {
                            board.play(cozy_move);
                            history_hashes.push(board.hash());
                        }
                    }
                }
                UciMessage::Go { time_control, search_control } => {
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Some(handle) = search_thread.take() {
                        let _ = handle.join();
                    }

                    stop_flag = Arc::new(AtomicBool::new(false));
                    let stop_clone = Arc::clone(&stop_flag);
                    
                    let ponder = line.contains("ponder");
                    is_pondering.store(ponder, Ordering::Relaxed);
                    let ponder_clone = Arc::clone(&is_pondering);

                    let board_clone = board.clone();
                    let history_clone = history_hashes.clone();
                    let tt_clone = Arc::clone(&tt);
                    let current_time_constants = time_constants.clone(); 
                    let current_search_constants = search_constants.clone();

                    let mut time_left = None;
                    let mut increment = None;
                    let mut moves_to_go_val = None;
                    let mut m_time = None;
                    let mut depth = MAX_SEARCH_DEPTH;

                    if let Some(tc) = time_control {
                        match tc {
                            vampirc_uci::UciTimeControl::TimeLeft { white_time, black_time, white_increment, black_increment, moves_to_go } => {
                                let (time, inc) = if board.side_to_move() == cozy_chess::Color::White {
                                    (white_time, white_increment)
                                } else {
                                    (black_time, black_increment)
                                };
                                
                                time_left = time.and_then(|t| t.to_std().ok());
                                increment = inc.and_then(|i| i.to_std().ok());
                                moves_to_go_val = moves_to_go.map(|m| m as u32);
                            }
                            vampirc_uci::UciTimeControl::MoveTime(t) => {
                                m_time = t.to_std().ok();
                            }
                            _ => {}
                        }
                    }

                    let mut limit = SearchLimit::from_uci(time_left, increment, moves_to_go_val, m_time, current_time_constants);

                    if let Some(sc) = search_control {
                        if let Some(d) = sc.depth {
                            depth = d;
                            if limit == SearchLimit::Infinite {
                                limit = SearchLimit::Depth(d as usize);
                            }
                        }
                        if let Some(n) = sc.nodes {
                            limit = SearchLimit::Nodes(n);
                        }
                    }

                    search_thread = Some(std::thread::spawn(move || {
                        if board_clone.status() != GameStatus::Ongoing {
                            println!("bestmove (none)");
                            return;
                        }

                        let best_move = find_best_move(
                            &board_clone,
                            depth,
                            limit,
                            &history_clone,
                            &tt_clone,
                            stop_clone,
                            ponder_clone,
                            current_search_constants
                        );

                        if let Some(m) = best_move {
                            let mut move_str = m.to_string();
                            
                            if board_clone.piece_on(m.from) == Some(Piece::King) {
                                match move_str.as_str() {
                                    "e1h1" => move_str = "e1g1".to_string(),
                                    "e1a1" => move_str = "e1c1".to_string(),
                                    "e8h8" => move_str = "e8g8".to_string(),
                                    "e8a8" => move_str = "e8c8".to_string(),
                                    _ => {}
                                }
                            }
                            
                            println!("bestmove {}", move_str);
                        } else {
                            println!("bestmove (none)");
                        }
                    }));
                }
                UciMessage::PonderHit => {
                    is_pondering.store(false, Ordering::Relaxed);
                }
                UciMessage::Stop => {
                    stop_flag.store(true, Ordering::Relaxed);
                    is_pondering.store(false, Ordering::Relaxed);
                }
                UciMessage::Quit => {
                    stop_flag.store(true, Ordering::Relaxed);
                    std::process::exit(0);
                }
                _ => {}
            }
        }
    }
}
