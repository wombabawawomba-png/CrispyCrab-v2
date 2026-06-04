// src/search.rs

use cozy_chess::{BitBoard, Board, Color, File, Move, Piece, Rank, Square};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;
use std::cell::RefCell;

use crate::time::{SearchLimit, TimeManager};
use crate::weigh::{PawnCache, LocalEvalCache, SearchBoard, evaluate, SEE_VALS};

// ============================================================================
//  GLOBAL STATICS & PRECOMPUTED TABLES
// ============================================================================
pub static NODES: AtomicU64 = AtomicU64::new(0);
pub static TT_AGE: AtomicU8 = AtomicU8::new(0);

pub const MAX_SEARCH_DEPTH: u8 = 64;
pub const MAX_QUIESCENCE_DEPTH: u8 = 12;

pub const MATE_SCORE: i32 = 30000;
pub const INFINITY: i32 = 32000;
pub const DRAW_SCORE: i32 = 0;

// LMP table optimized for tactical safety
const LMP_LIMITS: [[i32; 9]; 2] = [
    // improving = false
    [2, 3, 5, 8, 12, 17, 24, 32, 42],
    // improving = true
    [3, 4, 7, 11, 16, 23, 31, 41, 53],
];

// Global Precomputed LMR Table (Zero runtime per-thread compute)
pub static LMR_TABLE: LazyLock<[[u8; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0u8; 64]; 64];
    let base = 0.75;
    let div = 2.25;
    for d in 0..64 {
        for i in 0..64 {
            if d < 3 || i < 2 { table[d][i] = 0; }
            else { table[d][i] = (base + (d as f64).ln() * (i as f64).ln() / div) as u8; }
        }
    }
    table
});

// ============================================================================
//  THREAD-LOCAL HISTORY TABLES (Optimized: No Atomics, No Cache Contention)
// ============================================================================
pub struct LocalHistory {
    pub main: Box<[[[i16; 64]; 64]; 2]>,
    pub capture: Box<[[[i16; 64]; 6]; 6]>,
    pub pawn: Box<[[[i16; 64]; 64]; 2]>,
    pub minor: Box<[[[i16; 64]; 64]; 2]>,
    pub continuation: Box<[i16; 147456 * 2]>,
}

impl LocalHistory {
    pub fn new() -> Self {
        let main = Box::new([[[0i16; 64]; 64]; 2]);
        let capture = Box::new([[[0i16; 64]; 6]; 6]);
        let pawn = Box::new([[[0i16; 64]; 64]; 2]);
        let minor = Box::new([[[0i16; 64]; 64]; 2]);
        
        let continuation = vec![0i16; 147456 * 2]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to allocate continuation history"));

        Self { main, capture, pawn, minor, continuation }
    }

    pub fn clear(&mut self) {
        *self.main = [[[0i16; 64]; 64]; 2];
        *self.capture = [[[0i16; 64]; 6]; 6];
        *self.pawn = [[[0i16; 64]; 64]; 2];
        *self.minor = [[[0i16; 64]; 64]; 2];
        self.continuation.fill(0);
    }

    pub fn decay(&mut self) {
        for c in 0..2 {
            for f in 0..64 {
                for t in 0..64 {
                    self.main[c][f][t] /= 2;
                    self.pawn[c][f][t] /= 2;
                    self.minor[c][f][t] /= 2;
                }
            }
        }
        for a in 0..6 {
            for v in 0..6 {
                for t in 0..64 {
                    self.capture[a][v][t] /= 2;
                }
            }
        }
        for val in self.continuation.iter_mut() {
            *val /= 2;
        }
    }

    // Static function to prevent borrow checker conflicts (E0499)
    #[inline(always)]
    fn update_val(val_ref: &mut i16, bonus: i32) {
        let val = *val_ref as i32;
        *val_ref = (val + bonus - (val * bonus.abs()) / 16384).clamp(-16384, 16384) as i16;
    }

    #[inline(always)]
    fn update_main(&mut self, color: usize, from: usize, to: usize, bonus: i32) {
        let val_ref = &mut self.main[color][from][to];
        Self::update_val(val_ref, bonus);
    }

    #[inline(always)]
    fn update_capture(&mut self, attacker: usize, victim: usize, to: usize, bonus: i32) {
        let val_ref = &mut self.capture[attacker][victim][to];
        Self::update_val(val_ref, bonus);
    }

    #[inline(always)]
    fn update_pawn(&mut self, color: usize, from: usize, to: usize, bonus: i32) {
        let val_ref = &mut self.pawn[color][from][to];
        Self::update_val(val_ref, bonus);
    }

    #[inline(always)]
    fn update_minor(&mut self, color: usize, from: usize, to: usize, bonus: i32) {
        let val_ref = &mut self.minor[color][from][to];
        Self::update_val(val_ref, bonus);
    }

    #[inline(always)]
    fn update_cont(&mut self, ply_idx: usize, p1: usize, to1: usize, p2: usize, to2: usize, bonus: i32) {
        let idx = ply_idx * 147456 + p1 * 24576 + to1 * 384 + p2 * 64 + to2;
        let val_ref = &mut self.continuation[idx];
        Self::update_val(val_ref, bonus);
    }

    #[inline(always)]
    fn get_main(&self, color: usize, from: usize, to: usize) -> i32 {
        self.main[color][from][to] as i32
    }

    #[inline(always)]
    fn get_capture(&self, attacker: usize, victim: usize, to: usize) -> i32 {
        self.capture[attacker][victim][to] as i32
    }

    #[inline(always)]
    fn get_pawn(&self, color: usize, from: usize, to: usize) -> i32 {
        self.pawn[color][from][to] as i32
    }

    #[inline(always)]
    fn get_minor(&self, color: usize, from: usize, to: usize) -> i32 {
        self.minor[color][from][to] as i32
    }

    #[inline(always)]
    fn get_cont(&self, ply_idx: usize, p1: usize, to1: usize, p2: usize, to2: usize) -> i32 {
        let idx = ply_idx * 147456 + p1 * 24576 + to1 * 384 + p2 * 64 + to2;
        self.continuation[idx] as i32
    }
}

// ============================================================================
//  COMPATIBILITY PLACEMARKER FOR src/uci.rs
// ============================================================================
pub struct DummyHistory;
impl DummyHistory {
    pub fn clear(&self) {
        clear_global_history();
    }
    pub fn decay(&self) {
        MAIN_THREAD_DATA.with(|td_cell| {
            td_cell.borrow_mut().history.decay();
        });
    }
}
pub static GLOBAL_HISTORY: DummyHistory = DummyHistory;

// ============================================================================
//  CUSTOM THREAD POOL FOR LAZY SMP (Scale-Friendly Thread Task Management)
// ============================================================================
#[derive(Clone)]
struct SearchTask {
    bd: Board,
    max_depth: u8,
    tm: TimeManager,
    hs: Arc<[u64]>, 
    tt: Arc<TranspositionTable>,
    stop_flag: Arc<AtomicBool>,
    is_pondering: Arc<AtomicBool>,
    search_constants: SearchConstants,
    run_id: u64,
}

static TASK_MUTEX: LazyLock<Mutex<Option<SearchTask>>> = LazyLock::new(|| Mutex::new(None));
static TASK_CONDVAR: LazyLock<Condvar> = LazyLock::new(|| Condvar::new());
static ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);
static POOL_INITIALIZED: AtomicBool = AtomicBool::new(false);

fn init_thread_pool() {
    if POOL_INITIALIZED.swap(true, Ordering::SeqCst) { return; }
    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1);
    
    for i in 1..num_threads {
        std::thread::spawn(move || {
            let mut td = ThreadData::new(i, SearchConstants::default());
            let mut last_run_id = 0;
            
            loop {
                let task = {
                    let mut lock = TASK_MUTEX.lock().unwrap();
                    while lock.is_none() || lock.as_ref().unwrap().run_id == last_run_id {
                        lock = TASK_CONDVAR.wait(lock).unwrap();
                    }
                    ACTIVE_WORKERS.fetch_add(1, Ordering::SeqCst);
                    lock.as_ref().unwrap().clone()
                };
                
                last_run_id = task.run_id;
                
                td.search_constants = task.search_constants.clone();
                td.clear_for_search(); 
                
                let mut local_tm = task.tm.clone();
                
                search_worker(
                    &task.bd, task.max_depth, &mut local_tm,
                    &task.hs, &task.tt, &task.stop_flag, &task.is_pondering, false, &mut td
                );
                
                ACTIVE_WORKERS.fetch_sub(1, Ordering::SeqCst);
            }
        });
    }
}

// ============================================================================
//  TUNABLE SEARCH CONSTANTS & EVALUATION
// ============================================================================
#[derive(Clone, Debug, PartialEq)]
pub struct SearchConstants {
    pub aspiration_delta: i32,
    pub max_delta: i32,
    pub max_delta_capture: i32,
    pub qs_futility_margin: i32,
    pub see_threshold: i32,
    pub razor_margin: i32,
    pub probcut_margin: i32,
    pub singular_depth_threshold: u8,
    pub singular_margin_multiplier: i32,
    pub tempo_bonus: i32,
    pub rfp_multiplier: i32,
    pub futility_multiplier: i32,
    pub see_quiet_multiplier: i32,
    pub history_pruning_multiplier: i32,
    pub lmr_base: f64,
    pub lmr_divisor: f64,
    
    // SPSA Tunable NMP Parameters (Parameterized original strong formula)
    pub nmp_base: i32,
    pub nmp_depth_div: i32,
    pub nmp_eval_div: i32,
    pub nmp_eval_limit: i32,
}

impl Default for SearchConstants {
    fn default() -> Self {
        Self {
            aspiration_delta: 32,
            max_delta: 320,
            max_delta_capture: 1220,
            qs_futility_margin: 130,
            see_threshold: -90,
            razor_margin: 280,
            probcut_margin: 220, 
            singular_depth_threshold: 7,
            singular_margin_multiplier: 2,
            tempo_bonus: 14,
            rfp_multiplier: 75,
            futility_multiplier: 150,
            see_quiet_multiplier: -200,
            history_pruning_multiplier: -4000,
            lmr_base: 0.7601,
            lmr_divisor: 2.2452,
            // Default Values matches exactly your strong original formula:
            // reduction = 3 + depth / 4 + clamp((eval - beta) / 200, 0, 3)
            nmp_base: 3,
            nmp_depth_div: 4,
            nmp_eval_div: 200,
            nmp_eval_limit: 3,
        }
    }
}

const DUMMY_MOVE: Move = Move { from: Square::A1, to: Square::A1, promotion: None };

pub struct ThreadData {
    pub killers: Box<[[Option<Move>; 2]; 128]>,
    pub counter_moves: Box<[[Option<Move>; 64]; 6]>,
    pub pawn_cache: PawnCache,
    pub eval_cache: LocalEvalCache,
    pub local_nodes: u64,
    pub thread_id: usize,
    pub poll_mask: u64, 
    pub search_constants: SearchConstants,
    pub history: LocalHistory, // Thread-local history table (No Atomics!)
}

impl ThreadData {
    fn new(thread_id: usize, search_constants: SearchConstants) -> Self {
        let poll_mask = 4095;

        Self {
            killers: Box::new([[None; 2]; 128]),
            counter_moves: Box::new([[None; 64]; 6]),
            pawn_cache: PawnCache::new(),
            eval_cache: LocalEvalCache::new(),
            local_nodes: 0,
            thread_id,
            poll_mask,
            search_constants,
            history: LocalHistory::new(),
        }
    }

    #[inline(always)]
    fn clear_for_search(&mut self) {
        for i in 0..128 {
            self.killers[i] = [None; 2];
        }
        for i in 0..6 {
            for j in 0..64 {
                self.counter_moves[i][j] = None;
            }
        }
        self.local_nodes = 0;
        // History decays on new search task to preserve positional long-term heuristic values
        self.history.decay();
    }
}

thread_local! {
    static MAIN_THREAD_DATA: RefCell<ThreadData> = RefCell::new(ThreadData::new(0, SearchConstants::default()));
}

pub fn clear_global_history() {
    MAIN_THREAD_DATA.with(|td_cell| {
        td_cell.borrow_mut().history.clear();
    });
}

// ============================================================================
//  TRANSPOSITION TABLE (2-Way Bucket Associative TT)
// ============================================================================
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TTNodeType { None = 0, Exact = 1, Alpha = 2, Beta = 3 }

struct TTEntry { key: AtomicU64, data: AtomicU64 }

pub struct TranspositionTable { table: Vec<TTEntry>, size: usize }

#[inline(always)]
fn score_to_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_SCORE - 1000 { score + ply as i32 }
    else if score <= -MATE_SCORE + 1000 { score - ply as i32 }
    else { score }
}

#[inline(always)]
fn score_from_tt(score: i32, ply: u8) -> i32 {
    if score >= MATE_SCORE - 1000 { score - ply as i32 }
    else if score <= -MATE_SCORE + 1000 { score + ply as i32 }
    else { score }
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let entry_size = std::mem::size_of::<TTEntry>(); 
        let desired_entries = bytes / entry_size;
        let desired_buckets = desired_entries / 2;

        let mut size = desired_buckets.next_power_of_two();
        if size > desired_buckets && size > 1 {
            size >>= 1;
        }
        if size == 0 { size = 1; }

        let mut table = Vec::with_capacity(size * 2);
        for _ in 0..(size * 2) { 
            table.push(TTEntry { key: AtomicU64::new(0), data: AtomicU64::new(0) }); 
        }
        Self { table, size }
    }
    
    pub fn clear(&self) {
        for entry in &self.table {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }

    fn pack(score: i32, depth: u8, node_type: TTNodeType, best_move: Option<Move>, age: u8) -> u64 {
        let mv_u16 = if let Some(m) = best_move {
            let f = m.from as u16;
            let t = m.to as u16;
            let p = match m.promotion {
                None => 0, Some(Piece::Knight) => 1, Some(Piece::Bishop) => 2,
                Some(Piece::Rook) => 3, Some(Piece::Queen) => 4, _ => 0,
            };
            f | (t << 6) | (p << 12)
        } else { u16::MAX };
        let score_i16 = score.clamp(-32767, 32767) as i16;
        let score_u16 = score_i16 as u16;
        (mv_u16 as u64) | ((score_u16 as u64) << 16) | ((depth as u64) << 32) | ((node_type as u64) << 40) | ((age as u64) << 48)
    }
    
    fn unpack(data: u64) -> (i32, u8, TTNodeType, Option<Move>, u8) {
        if data == 0 { return (0, 0, TTNodeType::None, None, 0); }
        let mv_u16 = (data & 0xFFFF) as u16;
        let score_u16 = ((data >> 16) & 0xFFFF) as u16;
        let score = (score_u16 as i16) as i32;
        let depth = ((data >> 32) & 0xFF) as u8;
        let type_raw = ((data >> 40) & 0x3) as u8;
        let age = ((data >> 48) & 0xFF) as u8;
        let node_type = match type_raw {
            1 => TTNodeType::Exact, 2 => TTNodeType::Alpha, 3 => TTNodeType::Beta, _ => TTNodeType::None,
        };
        let best_move = if mv_u16 == u16::MAX { None } else {
            let f = mv_u16 & 0x3F;
            let t = (mv_u16 >> 6) & 0x3F;
            let p = match (mv_u16 >> 12) & 0x7 {
                1 => Some(Piece::Knight), 2 => Some(Piece::Bishop),
                3 => Some(Piece::Rook), 4 => Some(Piece::Queen), _ => None,
            };
            Some(Move { from: Square::index(f as usize), to: Square::index(t as usize), promotion: p })
        };
        (score, depth, node_type, best_move, age)
    }

    pub fn probe(&self, hash: u64, ply: u8) -> (bool, i32, u8, TTNodeType, Option<Move>) {
        let bucket_idx = (hash as usize) & (self.size - 1);
        let idx0 = bucket_idx * 2;
        let idx1 = idx0 + 1;

        for idx in [idx0, idx1] {
            let entry = &self.table[idx];
            let key_xor_data = entry.key.load(Ordering::Acquire);
            let data = entry.data.load(Ordering::Acquire);
            
            if key_xor_data ^ data == hash {
                let (mut score, depth, node_type, best_move, _) = Self::unpack(data);
                score = score_from_tt(score, ply);
                return (true, score, depth, node_type, best_move);
            }
        }
        (false, 0, 0, TTNodeType::None, None)
    }

    pub fn store(&self, hash: u64, mut score: i32, depth: u8, node_type: TTNodeType, best_move: Option<Move>, ply: u8) {
        score = score_to_tt(score, ply);
        let bucket_idx = (hash as usize) & (self.size - 1);
        let current_age = TT_AGE.load(Ordering::Relaxed);
        let packed = Self::pack(score, depth, node_type, best_move, current_age);

        let idx0 = bucket_idx * 2;
        let idx1 = idx0 + 1;

        let entry0 = &self.table[idx0];
        let entry1 = &self.table[idx1];

        let d0 = entry0.data.load(Ordering::Relaxed);
        let k0 = entry0.key.load(Ordering::Relaxed);
        let matches0 = k0 ^ d0 == hash;

        let d1 = entry1.data.load(Ordering::Relaxed);
        let k1 = entry1.key.load(Ordering::Relaxed);
        let matches1 = k1 ^ d1 == hash;

        if matches0 {
            let (_, current_depth, _, _, entry_age) = Self::unpack(d0);
            if depth >= current_depth || entry_age != current_age {
                entry0.data.store(packed, Ordering::Release);
                entry0.key.store(hash ^ packed, Ordering::Release);
            }
        } else if matches1 {
            let (_, current_depth, _, _, entry_age) = Self::unpack(d1);
            if depth >= current_depth || entry_age != current_age {
                entry1.data.store(packed, Ordering::Release);
                entry1.key.store(hash ^ packed, Ordering::Release);
            }
        } else {
            let (_, dep0, _, _, age0) = Self::unpack(d0);
            let (_, dep1, _, _, age1) = Self::unpack(d1);

            let score0 = (age0 == current_age) as i32 * 1000 + dep0 as i32;
            let score1 = (age1 == current_age) as i32 * 1000 + dep1 as i32;

            if score0 <= score1 {
                entry0.data.store(packed, Ordering::Release);
                entry0.key.store(hash ^ packed, Ordering::Release);
            } else {
                entry1.data.store(packed, Ordering::Release);
                entry1.key.store(hash ^ packed, Ordering::Release);
            }
        }
    }
}

// ============================================================================
//  MEMORY PREFETCHING HELPERS
// ============================================================================
#[inline(always)]
fn prefetch_tt(tt: &TranspositionTable, hash: u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let bucket_idx = (hash as usize) & (tt.size - 1);
        let ptr = tt.table.as_ptr().add(bucket_idx * 2) as *const i8;
        core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
    }
}

// ============================================================================
//  SEARCH HELPERS & FAST SCORING
// ============================================================================
#[inline(always)]
fn get_smallest_attacker(bd: &Board, sq: Square, occ: BitBoard, color: Color) -> Option<(Square, Piece)> {
    let us = bd.colors(color);
    let pawns = bd.pieces(Piece::Pawn) & us;
    let pawn_attacks = cozy_chess::get_pawn_attacks(sq, !color) & pawns & occ;
    if !pawn_attacks.is_empty() { return Some((pawn_attacks.into_iter().next().unwrap(), Piece::Pawn)); }
    let knights = bd.pieces(Piece::Knight) & us;
    let knight_attacks = cozy_chess::get_knight_moves(sq) & knights & occ;
    if !knight_attacks.is_empty() { return Some((knight_attacks.into_iter().next().unwrap(), Piece::Knight)); }
    let bishops = bd.pieces(Piece::Bishop) & us;
    let bishop_attacks = cozy_chess::get_bishop_moves(sq, occ) & bishops & occ;
    if !bishop_attacks.is_empty() { return Some((bishop_attacks.into_iter().next().unwrap(), Piece::Bishop)); }
    let rooks = bd.pieces(Piece::Rook) & us;
    let rook_attacks = cozy_chess::get_rook_moves(sq, occ) & rooks & occ;
    if !rook_attacks.is_empty() { return Some((rook_attacks.into_iter().next().unwrap(), Piece::Rook)); }
    let queens = bd.pieces(Piece::Queen) & us;
    let queen_attacks = (cozy_chess::get_bishop_moves(sq, occ) | cozy_chess::get_rook_moves(sq, occ)) & queens & occ;
    if !queen_attacks.is_empty() { return Some((queen_attacks.into_iter().next().unwrap(), Piece::Queen)); }
    let kings = bd.pieces(Piece::King) & us;
    let king_attacks = cozy_chess::get_king_moves(sq) & kings & occ;
    if !king_attacks.is_empty() { return Some((king_attacks.into_iter().next().unwrap(), Piece::King)); }
    None
}

fn see(bd: &Board, m: Move) -> i32 {
    let piece = bd.piece_on(m.from).unwrap_or(Piece::Pawn);
    let is_castling = piece == Piece::King && bd.color_on(m.to) == Some(bd.side_to_move());
    if is_castling { return 0; }

    let mut gain = [0; 32];
    let mut d = 0;
    let mut occ = bd.occupied();
    let mut color = bd.side_to_move();
    let target_piece = bd.piece_on(m.to);
    gain[d] = if let Some(p) = target_piece { SEE_VALS[p as usize] } else { 0 };
    if let Some(prom) = m.promotion { gain[d] += SEE_VALS[prom as usize] - SEE_VALS[Piece::Pawn as usize]; } 
    else if target_piece.is_none() && piece == Piece::Pawn && m.from.file() != m.to.file() {
        gain[d] = SEE_VALS[Piece::Pawn as usize];
        let ep_sq = Square::new(m.to.file(), m.from.rank());
        occ ^= BitBoard(1u64 << ep_sq as usize);
    }
    let mut current_piece = if let Some(prom) = m.promotion { prom } else { piece };
    occ ^= BitBoard(1u64 << m.from as usize);
    occ |= BitBoard(1u64 << m.to as usize);
    color = !color;
    loop {
        d += 1;
        if d >= 32 { break; }
        if let Some((sq, p)) = get_smallest_attacker(bd, m.to, occ, color) {
            occ ^= BitBoard(1u64 << sq as usize);
            gain[d] = SEE_VALS[current_piece as usize] - gain[d - 1];
            current_piece = p;
            color = !color;
        } else { break; }
    }
    d -= 1;
    while d > 0 { gain[d - 1] = -(-gain[d - 1]).max(gain[d]); d -= 1; }
    gain[0]
}

#[inline(always)]
fn is_legal_move(bd: &Board, m: Move) -> bool {
    bd.is_legal(m)
}

fn extract_pv(bd: &Board, tt: &TranspositionTable, depth: u8) -> String {
    let mut pv = String::new();
    let mut current_bd = bd.clone();
    let mut path = Vec::with_capacity(depth as usize);

    for _ in 0..depth {
        let hash = current_bd.hash();
        if path.contains(&hash) { break; } 
        path.push(hash);

        let (found, _, _, _, m_opt) = tt.probe(hash, 0);
        if found {
            if let Some(m) = m_opt {
                if is_legal_move(&current_bd, m) {
                    if !pv.is_empty() { pv.push(' '); }
                    pv.push_str(&m.to_string());
                    current_bd.play(m);
                    continue;
                }
            }
        }
        break;
    }
    pv
}

#[inline(always)]
fn score_capture(_bd: &Board, m: Move, history: &LocalHistory, cached_attacker: Piece, cached_victim: Option<Piece>) -> i32 {
    let mut score = 0;
    if let Some(cap_p) = cached_victim {
        let mvv_lva = SEE_VALS[cap_p as usize] * 16 - SEE_VALS[cached_attacker as usize];
        let hist = history.get_capture(cached_attacker as usize, cap_p as usize, m.to as usize);
        score += 2_000_000 + mvv_lva + hist / 8;
    }
    if let Some(prom) = m.promotion {
        let prom_val = SEE_VALS[prom as usize];
        if prom == Piece::Queen {
            score += 2_500_000 + prom_val;
        } else {
            score += 1_500_000 + prom_val;
        }
    }
    score
}

// Optimized Pass-By-Reference for prev_moves array parameter
#[inline(always)]
fn score_quiet(bd: &Board, m: Move, td: &ThreadData, prev_moves: &[Option<(Piece, Square)>; 6], ply: u8) -> i32 {
    let color = bd.side_to_move();
    let piece = bd.piece_on(m.from).unwrap_or(Piece::Pawn);
    let ply_idx = (ply as usize).min(127);

    if td.killers[ply_idx][0] == Some(m) { return 900_000; }
    if td.killers[ply_idx][1] == Some(m) { return 890_000; }

    if let Some((prev_p, prev_sq)) = prev_moves[0] {
        if td.counter_moves[prev_p as usize][prev_sq as usize] == Some(m) { return 800_000; }
    }

    let history = &td.history;
    let mut score = history.get_main(color as usize, m.from as usize, m.to as usize);
    if piece == Piece::Pawn {
        score += history.get_pawn(color as usize, m.from as usize, m.to as usize);
    } else if piece == Piece::Knight || piece == Piece::Bishop {
        score += history.get_minor(color as usize, m.from as usize, m.to as usize);
    }
    
    for i in 0..2 {
        if let Some((prev_p, prev_sq)) = prev_moves[i] {
            score += history.get_cont(i, prev_p as usize, prev_sq as usize, piece as usize, m.to as usize);
        } else { break; }
    }
    score.clamp(-100_000, 100_000)
}

#[inline(always)]
fn is_zugzwang_risk(bd: &Board) -> bool {
    let us = bd.colors(bd.side_to_move());
    let non_pawns = (bd.pieces(Piece::Knight) | bd.pieces(Piece::Bishop) | bd.pieces(Piece::Rook) | bd.pieces(Piece::Queen)) & us;
    non_pawns.is_empty()
}

// ============================================================================
//  LAZY/STAGED MOVE PICKER (Uses Thread-Local Non-Atomic History Lookup)
// ============================================================================
#[derive(PartialEq)]
enum PickerStage { TTMove, GenerateCaptures, YieldCaptures, GenerateQuiets, YieldQuiets, Done }

struct MovePicker {
    stage: PickerStage,
    tt_move: Option<Move>,
    excluded_move: Option<Move>,
    moves: [Move; 256],
    scores: [i32; 256],
    len: usize,
    idx: usize,
    qs_only: bool,
}

impl MovePicker {
    #[inline(always)]
    fn new(tt_move: Option<Move>, qs_only: bool, excluded_move: Option<Move>) -> Self {
        Self {
            stage: PickerStage::TTMove,
            tt_move,
            excluded_move,
            moves: [DUMMY_MOVE; 256],
            scores: [0; 256],
            len: 0,
            idx: 0,
            qs_only,
        }
    }

    #[inline(always)]
    fn next(
        &mut self, 
        bd: &Board, 
        td: &ThreadData, 
        prev_moves: &[Option<(Piece, Square)>; 6], 
        ply: u8
    ) -> Option<Move> {
        loop {
            match self.stage {
                PickerStage::TTMove => {
                    self.stage = PickerStage::GenerateCaptures;
                    if let Some(m) = self.tt_move {
                        if Some(m) != self.excluded_move {
                            return Some(m);
                        }
                    }
                }
                PickerStage::GenerateCaptures => {
                    self.len = 0; self.idx = 0;
                    let them = bd.colors(!bd.side_to_move());
                    let ep_sq = bd.en_passant().map(|f| Square::new(f, if bd.side_to_move() == Color::White { Rank::Sixth } else { Rank::Third }));
                    
                    let checkers = bd.checkers();
                    let num_checkers = checkers.len();

                    bd.generate_moves(|mut pm| {
                        if num_checkers > 1 && pm.piece != Piece::King {
                            return false;
                        }

                        let mut_cap_mask = them;
                        let mut cap_mask = mut_cap_mask;
                        if let Some(ep) = ep_sq { cap_mask |= BitBoard(1u64 << ep as usize); }
                        if pm.piece == Piece::Pawn {
                            let promo_rank = if bd.side_to_move() == Color::White { BitBoard(0xFF00000000000000) } else { BitBoard(0x00000000000000FF) };
                            pm.to &= cap_mask | promo_rank;
                        } else { pm.to &= cap_mask; }
                        
                        for m in pm {
                            if Some(m) != self.tt_move && Some(m) != self.excluded_move {
                                let attacker = pm.piece;
                                let victim = bd.piece_on(m.to);
                                let is_ep = victim.is_none() && attacker == Piece::Pawn && m.from.file() != m.to.file();
                                let final_victim = if is_ep { Some(Piece::Pawn) } else { victim };
                                
                                self.moves[self.len] = m;
                                
                                let see_val = see(bd, m);
                                if see_val >= 0 {
                                    self.scores[self.len] = score_capture(bd, m, &td.history, attacker, final_victim) + see_val;
                                } else {
                                    self.scores[self.len] = 100_000 + see_val + SEE_VALS[final_victim.unwrap_or(Piece::Pawn) as usize];
                                }
                                self.len += 1;
                            }
                        }
                        false
                    });
                    self.stage = PickerStage::YieldCaptures;
                }
                PickerStage::YieldCaptures => {
                    if self.idx < self.len {
                        let best_idx = self.select_best();
                        let m = self.moves[best_idx];
                        self.idx += 1;
                        return Some(m);
                    } else {
                        self.stage = if self.qs_only { PickerStage::Done } else { PickerStage::GenerateQuiets };
                    }
                }
                PickerStage::GenerateQuiets => {
                    self.len = 0; self.idx = 0;
                    let empty = !bd.occupied();
                    let checkers = bd.checkers();
                    let num_checkers = checkers.len();

                    bd.generate_moves(|mut pm| {
                        if num_checkers > 1 && pm.piece != Piece::King {
                            return false;
                        }

                        if pm.piece == Piece::Pawn {
                            let promo_rank = if bd.side_to_move() == Color::White { BitBoard(0xFF00000000000000) } else { BitBoard(0x00000000000000FF) };
                            pm.to &= empty & !promo_rank; 
                        } else if pm.piece == Piece::King {
                            let rooks = bd.pieces(Piece::Rook) & bd.colors(bd.side_to_move());
                            pm.to &= empty | rooks;
                        } else { pm.to &= empty; }
                        
                        for m in pm {
                            if Some(m) != self.tt_move && Some(m) != self.excluded_move {
                                self.moves[self.len] = m;
                                self.scores[self.len] = score_quiet(bd, m, td, prev_moves, ply);
                                self.len += 1;
                            }
                        }
                        false
                    });
                    self.stage = PickerStage::YieldQuiets;
                }
                PickerStage::YieldQuiets => {
                    if self.idx < self.len {
                        let best_idx = self.select_best();
                        let m = self.moves[best_idx];
                        self.idx += 1;
                        return Some(m);
                    } else { self.stage = PickerStage::Done; }
                }
                PickerStage::Done => return None,
            }
        }
    }

    #[inline(always)]
    fn select_best(&mut self) -> usize {
        let mut best_score = i32::MIN;
        let mut best_idx = self.idx;
        for i in self.idx..self.len {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }
        self.moves.swap(self.idx, best_idx);
        self.scores.swap(self.idx, best_idx);
        self.idx
    }

    #[inline(always)]
    pub fn get_see(&mut self, bd: &Board, m: Move) -> i32 {
        see(bd, m)
    }
}

// ============================================================================
//  QUIESCENCE SEARCH
// ============================================================================
pub fn quiescence<const PV: bool>(
    bd: &mut SearchBoard, mut alpha: i32, beta: i32, qd: u8, 
    stop_flag: &AtomicBool, td: &mut ThreadData, ply: u8, 
    tt: &Arc<TranspositionTable>
) -> i32 {
    if td.local_nodes & td.poll_mask == 0 {
        NODES.fetch_add(td.local_nodes, Ordering::Relaxed);
        td.local_nodes = 0;
        if stop_flag.load(Ordering::Relaxed) { return 0; }
    }
    td.local_nodes += 1;
    
    let hash = bd.cozy.hash();
    let mut qs_ttm = None;
    let (mut found, entry_score, _, entry_type, entry_move) = tt.probe(hash, ply);
    
    if found {
        if let Some(m) = entry_move {
            if is_legal_move(&bd.cozy, m) { qs_ttm = Some(m); } else { found = false; }
        }
    }

    if found {
        match entry_type {
            TTNodeType::Exact => return entry_score,
            TTNodeType::Alpha => if entry_score <= alpha { return entry_score; },
            TTNodeType::Beta => if entry_score >= beta { return entry_score; },
            _ => {}
        }
    }

    let in_check = !bd.cozy.checkers().is_empty();
    let stand_pat = evaluate(bd, hash, td);
    
    if qd == 0 { return stand_pat; }

    let mut best_score = if in_check { -MATE_SCORE + ply as i32 } else { stand_pat };

    if !in_check {
        if stand_pat >= beta { return beta; }
        if stand_pat + td.search_constants.max_delta_capture < alpha { return stand_pat; } 
        if stand_pat > alpha { alpha = stand_pat; }
    }

    let original_alpha = alpha;
    let prev_moves_dummy = [None; 6];
    let mut picker = MovePicker::new(qs_ttm, !in_check, None);
    let mut moves_searched = 0;

    while let Some(m) = picker.next(&bd.cozy, td, &prev_moves_dummy, ply) {
        if !in_check {
            let see_val = picker.get_see(&bd.cozy, m);
            if see_val < 0 { continue; } 

            let mut max_gain = 0;
            if let Some(prom) = m.promotion {
                max_gain += SEE_VALS[prom as usize] - SEE_VALS[Piece::Pawn as usize];
            }
            let captured_piece = bd.cozy.piece_on(m.to);
            let is_ep = captured_piece.is_none() && bd.cozy.piece_on(m.from) == Some(Piece::Pawn) && m.from.file() != m.to.file();
            let actual_captured = if is_ep { Some(Piece::Pawn) } else { captured_piece };
            if let Some(cap) = actual_captured {
                max_gain += SEE_VALS[cap as usize];
            }
            
            if stand_pat + max_gain + 300 < alpha {
                continue; 
            }
        }

        let mut nb = bd.clone();
        nb.play(m);
        moves_searched += 1;

        let s = -quiescence::<PV>(&mut nb, -beta, -alpha, qd - 1, stop_flag, td, ply + 1, tt);
        if s > best_score { best_score = s; }
        if s > alpha { alpha = s; }
        if alpha >= beta { break; }
    }

    if moves_searched == 0 && in_check { return -MATE_SCORE + ply as i32; }
    
    let node_type = if best_score >= beta { TTNodeType::Beta } 
                    else if best_score > original_alpha { TTNodeType::Exact } 
                    else { TTNodeType::Alpha };
                    
    tt.store(hash, best_score, 0, node_type, None, ply);
    
    best_score
}

// ============================================================================
//  NEGAMAX
// ============================================================================
pub fn negamax<const PV: bool>(
    bd: &mut SearchBoard, mut d: u8, mut alpha: i32, mut beta: i32,
    td: &mut ThreadData, tm: &TimeManager, is_pondering: &AtomicBool, hs: &[u64], 
    search_stack: &mut [u64; 256], eval_stack: &mut [i32; 256], tt: &Arc<TranspositionTable>,
    ply: u8, allow_nmp: bool, prev_moves: &[Option<(Piece, Square)>; 6], stop_flag: &AtomicBool,
    excluded_move: Option<Move>
) -> i32 {
    if td.local_nodes & td.poll_mask == 0 {
        NODES.fetch_add(td.local_nodes, Ordering::Relaxed);
        td.local_nodes = 0;
        if stop_flag.load(Ordering::Relaxed) || (!is_pondering.load(Ordering::Relaxed) && tm.is_hard_stop()) {
            stop_flag.store(true, Ordering::Relaxed);
            return 0;
        }
    }
    td.local_nodes += 1;

    if ply >= 128 {
        return evaluate(bd, bd.cozy.hash(), td);
    }

    // Mate Distance Pruning
    let mated_score = -MATE_SCORE + ply as i32;
    let mate_score = MATE_SCORE - ply as i32;
    alpha = alpha.max(mated_score);
    beta = beta.min(mate_score - 1);
    if alpha >= beta { return alpha; }

    let hash = bd.cozy.hash();
    let in_path = search_stack[..ply as usize].iter().any(|&h| h == hash);
    
    let halfmoves = bd.cozy.halfmove_clock() as usize;

    let history_to_check = if halfmoves > 0 {
        let check_len = halfmoves.min(hs.len());
        &hs[hs.len() - check_len..]
    } else {
        &[]
    };
    let history_reps = history_to_check.iter().any(|&h| h == hash);

    if ply > 0 && (in_path || history_reps || halfmoves >= 100) { return DRAW_SCORE; }

    let in_check = !bd.cozy.checkers().is_empty();
    if d == 0 { 
        return quiescence::<PV>(bd, alpha, beta, MAX_QUIESCENCE_DEPTH, stop_flag, td, ply, tt); 
    }
    
    search_stack[ply as usize] = hash;

    let mut ttm = None;
    let (mut found, entry_score, entry_depth, entry_type, entry_move) = tt.probe(hash, ply);
    
    if found {
        if let Some(m) = entry_move {
            if is_legal_move(&bd.cozy, m) { ttm = Some(m); } else { found = false; }
        }
    }

    if found {
        let is_excluded = excluded_move.is_some() && entry_move == excluded_move;
        if !is_excluded && entry_depth >= d {
            match entry_type {
                TTNodeType::Exact => return entry_score,
                TTNodeType::Alpha => if entry_score <= alpha { return entry_score; },
                TTNodeType::Beta => if entry_score >= beta { return entry_score; },
                _ => {}
            }
        }
    }

    if d >= 4 && ttm.is_none() && excluded_move.is_none() { d -= 1; }

    let mut static_eval = evaluate(bd, hash, td);

    if found && excluded_move.is_none() {
        if entry_type == TTNodeType::Exact {
            static_eval = entry_score;
        } else if entry_type == TTNodeType::Beta && entry_score > static_eval {
            static_eval = entry_score;
        } else if entry_type == TTNodeType::Alpha && entry_score < static_eval {
            static_eval = entry_score;
        }
    }

    let improving = if ply >= 2 {
        static_eval > eval_stack[(ply - 2) as usize]
    } else {
        true
    };
    eval_stack[ply as usize] = static_eval;

    // RFP / Static Null Move Pruning
    if !in_check && d <= 5 && !found && excluded_move.is_none() && beta.abs() < MATE_SCORE - 1000 {
        let rfp_margin = td.search_constants.rfp_multiplier * (d as i32) + if improving { 0 } else { 35 * (d as i32) };
        if static_eval - rfp_margin >= beta { return static_eval - rfp_margin; }
    }

    // Razoring
    if !in_check && d <= 3 && !found && excluded_move.is_none() {
        let razor_margin = alpha - td.search_constants.razor_margin - (d as i32 * 100) + if improving { 50 } else { 0 }; 
        if static_eval <= razor_margin {
            let q_score = quiescence::<PV>(bd, alpha, beta, MAX_QUIESCENCE_DEPTH, stop_flag, td, ply, tt);
            if q_score <= alpha { return q_score; }
        }
    }

    let is_zugzwang = is_zugzwang_risk(&bd.cozy);

    let mut nmp_threat = false;
    
    // ========================================================================
    //   NULL MOVE PRUNING FORMULA (Optimized Dynamic SPSA-Tunable Parameters)
    // ========================================================================
    if allow_nmp && d >= 3 && !in_check && static_eval >= beta && beta < MATE_SCORE - 1000 && !is_zugzwang && excluded_move.is_none() {
        // Prevents division-by-zero during randomized SPSA trials
        let depth_div = td.search_constants.nmp_depth_div.max(1);
        let eval_div = td.search_constants.nmp_eval_div.max(1);
        
        let depth_term = (d as i32) / depth_div;
        
        // Balanced evaluation scaling clamped to prevent over-reduction
        let eval_term = ((static_eval - beta) / eval_div)
            .clamp(0, td.search_constants.nmp_eval_limit);
            
        let r = td.search_constants.nmp_base + depth_term + eval_term;
        let reduction = r.max(1) as u8;

        if let Some(null_cozy) = bd.cozy.null_move() {
            let mut null_bd = SearchBoard {
                cozy: null_cozy,
                base_mg: bd.base_mg,
                base_eg: bd.base_eg,
                phase: bd.phase,
            };
            let mut null_prev_moves = [None; 6];
            for i in 1..6 { null_prev_moves[i] = prev_moves[i - 1]; }
            
            // Pass null_prev_moves by reference to avoid copying 48 bytes
            let null_score = -negamax::<false>(&mut null_bd, d.saturating_sub(1 + reduction), -beta, -beta + 1, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, false, &null_prev_moves, stop_flag, None);
            if null_score >= beta { return beta; } 
            else if null_score < alpha { nmp_threat = true; }
        }
    }

    if !PV && d >= 5 && beta.abs() < MATE_SCORE - 1000 && excluded_move.is_none() {
        let probcut_beta = beta + td.search_constants.probcut_margin; 
        let mut pc_picker = MovePicker::new(None, true, None); 
        while let Some(m) = pc_picker.next(&bd.cozy, td, prev_moves, ply) {
            if pc_picker.get_see(&bd.cozy, m) >= 0 {
                let mut nb = bd.clone();
                nb.play(m);
                let pc_depth = d - 3;
                let mut pc_prev_moves = [None; 6];
                pc_prev_moves[0] = Some((bd.cozy.piece_on(m.from).unwrap_or(Piece::Pawn), m.to));
                for i in 1..6 { pc_prev_moves[i] = prev_moves[i - 1]; }
                
                // Pass pc_prev_moves reference to negamax
                let s = -negamax::<false>(&mut nb, pc_depth, -probcut_beta, -probcut_beta + 1, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, false, &pc_prev_moves, stop_flag, None);
                if s >= probcut_beta {
                    tt.store(hash, s, pc_depth, TTNodeType::Beta, Some(m), ply);
                    return s;
                }
            }
        }
    }

    let mut singular_extension_level: u8 = 0;
    if excluded_move.is_none() && d >= td.search_constants.singular_depth_threshold && found && entry_depth >= d - 3 && entry_type != TTNodeType::Alpha && entry_score.abs() < MATE_SCORE - 1000 && ttm.is_some() {
        let singular_margin = (d as i32) * td.search_constants.singular_margin_multiplier; 
        let singular_beta = (entry_score - singular_margin).max(-MATE_SCORE + 1000);
        let r_depth = (d - 1) / 2;
        let s = negamax::<false>(bd, r_depth, singular_beta - 1, singular_beta, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply, false, prev_moves, stop_flag, ttm);
        if s < singular_beta - singular_margin {
            singular_extension_level = 2; 
        } else if s < singular_beta {
            singular_extension_level = 1; 
        }
    }

    let mut max_s = -INFINITY;
    let mut best_m = None;
    let mut moves_searched: i32 = 0;
    let original_alpha = alpha;
    let futility_margin = if improving {
        td.search_constants.futility_multiplier * (d as i32)
    } else {
        (td.search_constants.futility_multiplier - 30) * (d as i32)
    };
    let do_futility = d <= 4 && !in_check && !found && (static_eval + futility_margin < alpha);

    let mut searched_quiets = [DUMMY_MOVE; 64];
    let mut quiets_len = 0;
    
    let color = bd.cozy.side_to_move();
    let mut picker = MovePicker::new(ttm, false, excluded_move);

    while let Some(m) = picker.next(&bd.cozy, td, prev_moves, ply) {
        let piece = bd.cozy.piece_on(m.from).unwrap_or(Piece::Pawn);
        let victim = bd.cozy.piece_on(m.to);
        let is_castling = piece == Piece::King && bd.cozy.color_on(m.to) == Some(bd.cozy.side_to_move());
        let is_ep = victim.is_none() && piece == Piece::Pawn && m.from.file() != m.to.file();
        let is_capture = (victim.is_some() && !is_castling) || is_ep;
        let captured_piece = if is_ep { Some(Piece::Pawn) } else { victim };
        let is_promotion = m.promotion.is_some();
        let ply_idx = (ply as usize).min(127);
        let is_killer = td.killers[ply_idx][0] == Some(m) || td.killers[ply_idx][1] == Some(m);
        let is_advanced_pawn = piece == Piece::Pawn && (m.to.rank() == Rank::Seventh || m.to.rank() == Rank::Second);

        if !in_check && is_capture && !is_promotion && d <= 4 {
            if picker.get_see(&bd.cozy, m) < -50 * (d as i32) {
                continue;
            }
        }

        if d <= 4 && moves_searched > 0 && is_capture && !is_promotion && !in_check {
            let see_margin = td.search_constants.see_quiet_multiplier * (d as i32);
            if picker.get_see(&bd.cozy, m) < see_margin { continue; }
        }

        if moves_searched > 0 && !is_capture && !is_promotion {
            if d <= 5 && !is_killer && !in_check {
                let hist_score = td.history.get_main(color as usize, m.from as usize, m.to as usize);
                if hist_score < td.search_constants.history_pruning_multiplier * (d as i32) { continue; }
            }
            
            if d <= 8 && !in_check {
                let lmp_limit = LMP_LIMITS[improving as usize][d as usize];
                if moves_searched > lmp_limit && !is_killer && !is_advanced_pawn { continue; }
            }
            if do_futility && moves_searched > 1 && !is_killer && !is_advanced_pawn { continue; }
        }

        let mut nb = bd.clone();
        nb.play(m);
        
        prefetch_tt(tt, nb.cozy.hash());

        let gives_check = !nb.cozy.checkers().is_empty();

        let mut next_prev_moves = [None; 6];
        next_prev_moves[0] = Some((piece, m.to));
        for j in 1..6 { next_prev_moves[j] = prev_moves[j - 1]; }

        let extension = if Some(m) == ttm { singular_extension_level } else { 0 };
        let new_depth = d - 1 + extension;
        let mut s;

        if moves_searched == 0 {
            s = -negamax::<true>(&mut nb, new_depth, -beta, -alpha, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, true, &next_prev_moves, stop_flag, None);
        } else {
            let mut reduction = 0;
            if extension == 0 && d >= 2 && moves_searched >= 2 && !in_check && !is_capture && !is_promotion && !gives_check {
                let mut r = LMR_TABLE[(d as usize).min(63)][(moves_searched as usize).min(63)] as i32;
                
                if !improving { r += 1; }
                if nmp_threat { r -= 1; }
                if is_killer { r -= 1; }
                if is_zugzwang { r -= 1; }
                
                let hist_score = td.history.get_main(color as usize, m.from as usize, m.to as usize);
                let hist_modifier = (hist_score / 2048).clamp(-2, 2);
                let r_i32 = r - hist_modifier;

                reduction = r_i32.max(0) as u8;
            }

            let reduced_depth = new_depth.saturating_sub(reduction).max(0);
            s = -negamax::<false>(&mut nb, reduced_depth, -alpha - 1, -alpha, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, true, &next_prev_moves, stop_flag, None);

            if s > alpha && reduction > 0 {
                s = -negamax::<false>(&mut nb, new_depth, -alpha - 1, -alpha, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, true, &next_prev_moves, stop_flag, None);
            }
            if PV && s > alpha && s < beta {
                s = -negamax::<true>(&mut nb, new_depth, -beta, -alpha, td, tm, is_pondering, hs, search_stack, eval_stack, tt, ply + 1, true, &next_prev_moves, stop_flag, None);
            }
        }

        if stop_flag.load(Ordering::Relaxed) { return 0; }

        moves_searched += 1;
        if s > max_s { max_s = s; best_m = Some(m); }
        if s > alpha {
            alpha = s;
            if alpha >= beta {
                let bonus = ((d as i32) * (d as i32) + 2 * (d as i32)).min(450);

                if is_capture {
                    let cap_p = captured_piece.unwrap();
                    td.history.update_capture(piece as usize, cap_p as usize, m.to as usize, bonus);
                } else {
                    td.killers[ply_idx][1] = td.killers[ply_idx][0];
                    td.killers[ply_idx][0] = Some(m);

                    td.history.update_main(color as usize, m.from as usize, m.to as usize, bonus);
                    if piece == Piece::Pawn {
                        td.history.update_pawn(color as usize, m.from as usize, m.to as usize, bonus);
                    } else if piece == Piece::Knight || piece == Piece::Bishop {
                        td.history.update_minor(color as usize, m.from as usize, m.to as usize, bonus);
                    }

                    if let Some((prev_p, prev_sq)) = prev_moves[0] {
                        td.counter_moves[prev_p as usize][prev_sq as usize] = Some(m);
                    }

                    for j in 0..2 {
                        if let Some((prev_p, prev_sq)) = prev_moves[j] {
                            td.history.update_cont(j, prev_p as usize, prev_sq as usize, piece as usize, m.to as usize, bonus);
                        } else { break; }
                    }

                    for i in 0..quiets_len {
                        let qm = searched_quiets[i];
                        let qm_piece = bd.cozy.piece_on(qm.from).unwrap_or(Piece::Pawn);
                        td.history.update_main(color as usize, qm.from as usize, qm.to as usize, -bonus);
                        if qm_piece == Piece::Pawn {
                            td.history.update_pawn(color as usize, qm.from as usize, qm.to as usize, -bonus);
                        } else if qm_piece == Piece::Knight || qm_piece == Piece::Bishop {
                            td.history.update_minor(color as usize, qm.from as usize, qm.to as usize, -bonus);
                        }

                        for j in 0..2 {
                            if let Some((prev_p, prev_sq)) = prev_moves[j] {
                                    td.history.update_cont(j, prev_p as usize, prev_sq as usize, qm_piece as usize, qm.to as usize, -bonus);
                            } else { break; }
                        }
                    }
                }
                break;
            }
        }
        if !is_capture && !is_promotion {
            if s <= original_alpha && quiets_len < 64 {
                searched_quiets[quiets_len] = m;
                quiets_len += 1;
            }
        }
    }

    if moves_searched == 0 {
        return if excluded_move.is_some() { alpha } 
               else if in_check { -MATE_SCORE + ply as i32 } 
               else { DRAW_SCORE };
    }

    let node_type = if max_s >= beta { TTNodeType::Beta }
        else if max_s > original_alpha { TTNodeType::Exact }
        else { TTNodeType::Alpha };

    if excluded_move.is_none() {
        tt.store(hash, max_s, d, node_type, best_m, ply);
    }
    
    max_s
}

// ============================================================================
//  LAZY SMP WORKER
// ============================================================================
fn search_worker(
    bd: &Board, max_depth: u8, tm: &mut TimeManager,
    hs: &[u64], tt: &Arc<TranspositionTable>, 
    stop_flag: &Arc<AtomicBool>, is_pondering: &Arc<AtomicBool>, is_main: bool, td: &mut ThreadData
) -> Option<(Move, i32)> {
    let local_bd = SearchBoard::new(bd.clone());
    let mut root_moves_raw = Vec::new();
    local_bd.cozy.generate_moves(|pm| { root_moves_raw.extend(pm); false });

    if root_moves_raw.is_empty() { return None; }

    if is_main && root_moves_raw.len() == 1 {
        tm.notify_one_legal_move();
    }

    let mut root_moves: Vec<(Move, i32)> = root_moves_raw.into_iter().map(|m| (m, -INFINITY)).collect();
    let mut best_global = Some(root_moves[0].0);
    let mut last_score: i32 = 0;
    let mut d = 1;
    
    let mut search_stack = [0u64; 256];
    search_stack[0] = local_bd.cozy.hash();

    let mut eval_stack = [0i32; 256];
    eval_stack[0] = evaluate(&local_bd, local_bd.cozy.hash(), td);

    let mut was_pondering = is_pondering.load(Ordering::Relaxed);

    while d <= max_depth {
        if stop_flag.load(Ordering::Relaxed) { break; }

        let currently_pondering = is_pondering.load(Ordering::Relaxed);
        if was_pondering && !currently_pondering {
            tm.start(); 
            was_pondering = false;
        }

        let mut delta = td.search_constants.aspiration_delta; 
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;
        let mut fail_count = 0;
        
        let mut total_nodes_this_depth;
        let mut best_move_nodes;

        if d > 4 && last_score.abs() < MATE_SCORE - 1000 {
            delta = (20 + (d as i32) * 2).min(td.search_constants.max_delta);
            alpha = (last_score - delta).max(-INFINITY);
            beta = (last_score + delta).min(INFINITY);
        }

        let mut ttm = None;
        let (found, _, _, _, entry_move) = tt.probe(local_bd.cozy.hash(), 0);
        if found {
            if let Some(m) = entry_move {
                if is_legal_move(&local_bd.cozy, m) { ttm = Some(m); }
            }
        }

        let root_prev_moves = [None; 6];
        if d == 1 {
            for (m, score) in root_moves.iter_mut() {
                let piece = local_bd.cozy.piece_on(m.from).unwrap_or(Piece::Pawn);
                let victim = local_bd.cozy.piece_on(m.to);
                let is_castling = piece == Piece::King && victim == Some(Piece::Rook) && local_bd.cozy.color_on(m.to) == Some(local_bd.cozy.side_to_move());
                let is_ep = victim.is_none() && piece == Piece::Pawn && m.from.file() != m.to.file();
                let is_capture = (victim.is_some() && !is_castling) || is_ep;

                if is_capture {
                    let cap_p = if is_ep { Some(Piece::Pawn) } else { victim };
                    *score = score_capture(&local_bd.cozy, *m, &td.history, piece, cap_p);
                } else {
                    *score = score_quiet(&local_bd.cozy, *m, td, &root_prev_moves, 0);
                }
            }
            root_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        } else {
            root_moves.sort_by(|a, b| {
                if Some(a.0) == ttm { std::cmp::Ordering::Less }
                else if Some(b.0) == ttm { std::cmp::Ordering::Greater }
                else { b.1.cmp(&a.1) }
            });
        }

        if !is_main && root_moves.len() > 2 && d > 1 {
            let shift = td.thread_id % (root_moves.len() - 1);
            if shift > 0 { root_moves[1..].rotate_left(shift); }
        }

        let mut depth_aborted = false;
        loop {
            let mut best_move_this_iter = root_moves[0].0;
            let mut best_score_this_iter = -INFINITY;
            let mut search_alpha = alpha;
            
            total_nodes_this_depth = 0;
            best_move_nodes = 0;

            for (i, &mut (m, ref mut move_score)) in root_moves.iter_mut().enumerate() {
                if search_alpha >= beta { break; }
                if stop_flag.load(Ordering::Relaxed) { depth_aborted = true; break; }

                if !is_pondering.load(Ordering::Relaxed) && tm.is_hard_stop() {
                    stop_flag.store(true, Ordering::Relaxed);
                    depth_aborted = true;
                    break;
                }

                if let SearchLimit::Nodes(n) = tm.limit {
                    if NODES.load(Ordering::Relaxed) + td.local_nodes >= n {
                        stop_flag.store(true, Ordering::Relaxed);
                        depth_aborted = true;
                        break;
                    }
                }

                let nodes_before = td.local_nodes;
                let mut nb = local_bd.clone();
                nb.play(m);

                let piece = local_bd.cozy.piece_on(m.from).unwrap_or(Piece::Pawn);
                let mut next_prev_moves = [None; 6];
                next_prev_moves[0] = Some((piece, m.to));

                let s;
                if i == 0 {
                    s = -negamax::<true>(&mut nb, d - 1, -beta, -search_alpha, td, tm, is_pondering, hs, &mut search_stack, &mut eval_stack, tt, 1, true, &next_prev_moves, stop_flag, None);
                } else {
                    let mut temp_s = -negamax::<false>(&mut nb, d - 1, -search_alpha - 1, -search_alpha, td, tm, is_pondering, hs, &mut search_stack, &mut eval_stack, tt, 1, true, &next_prev_moves, stop_flag, None);
                    if temp_s > search_alpha && temp_s < beta && !stop_flag.load(Ordering::Relaxed) {
                        temp_s = -negamax::<true>(&mut nb, d - 1, -beta, -search_alpha, td, tm, is_pondering, hs, &mut search_stack, &mut eval_stack, tt, 1, true, &next_prev_moves, stop_flag, None);
                    }
                    s = temp_s;
                }

                let nodes_after = td.local_nodes;
                let nodes_spent = nodes_after - nodes_before;
                total_nodes_this_depth += nodes_spent;
                if i == 0 { best_move_nodes = nodes_spent; }

                if stop_flag.load(Ordering::Relaxed) { depth_aborted = true; break; }

                *move_score = s;
                if s > best_score_this_iter {
                    best_score_this_iter = s;
                    best_move_this_iter = m;
                    if s > search_alpha { search_alpha = s; }
                }
            }

            if depth_aborted { break; }

            if best_score_this_iter > alpha && best_score_this_iter < beta {
                last_score = best_score_this_iter;
                best_global = Some(best_move_this_iter);
                tt.store(local_bd.cozy.hash(), best_score_this_iter, d, TTNodeType::Exact, Some(best_move_this_iter), 0);

                NODES.fetch_add(td.local_nodes, Ordering::Relaxed);
                td.local_nodes = 0;

                if is_main {
                    let total_nodes = NODES.load(Ordering::Relaxed);
                    let elapsed = tm.start_time.elapsed();
                    let nps = if elapsed.as_micros() > 0 { (total_nodes as u128 * 1_000_000) / elapsed.as_micros() } else { 0 };
                    
                    let score_str = if last_score > MATE_SCORE - 100 {
                        format!("mate {}", (MATE_SCORE - last_score + 1) / 2)
                    } else if last_score < -MATE_SCORE + 100 {
                        format!("mate {}", (-MATE_SCORE - last_score - 1) / 2)
                    } else {
                        format!("cp {}", last_score)
                    };

                    let pv_str = extract_pv(&local_bd.cozy, tt, d);

                    println!("info depth {} score {} time {} nodes {} nps {} pv {}", 
                        d, score_str, elapsed.as_millis(), total_nodes, nps, pv_str);
                }
                break;
            }

            if let Some(idx) = root_moves.iter().position(|&(m, _)| m == best_move_this_iter) {
                root_moves.swap(0, idx);
            }

            if best_score_this_iter <= alpha {
                alpha = (alpha - delta).max(-INFINITY);
                delta += delta / 2; 
                fail_count += 1;
                if is_main {
                    tm.report_aspiration_fail(d as i32, TTNodeType::Alpha);
                }
            } else if best_score_this_iter >= beta {
                beta = (beta + delta).min(INFINITY);
                delta += delta / 2; 
                fail_count += 1;
            }

            if fail_count >= 3 {
                alpha = -INFINITY;
                beta = INFINITY;
                delta = td.search_constants.max_delta; 
            }
        }

        if depth_aborted { break; }
        if last_score >= MATE_SCORE - 1 { break; }

        if is_main && !is_pondering.load(Ordering::Relaxed) && matches!(tm.limit, SearchLimit::Dynamic { .. }) {
            let best_move_nodes_fraction = if total_nodes_this_depth > 0 {
                Some(best_move_nodes as f64 / total_nodes_this_depth as f64)
            } else { None };

            tm.report_completed_depth(d as i32, last_score, best_global.unwrap(), best_move_nodes_fraction);

            if tm.is_past_opt_time() {
                stop_flag.store(true, Ordering::Relaxed);
                break;
            }
        }

        d += 1;
    }
    NODES.fetch_add(td.local_nodes, Ordering::Relaxed);
    
    if is_main {
        while is_pondering.load(Ordering::Relaxed) && !stop_flag.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    if let Some(m) = best_global { Some((m, last_score)) } else { None }
}

// ============================================================================
//  ROOT SEARCH
// ============================================================================
pub fn find_best_move(
    bd: &Board, 
    max_depth: u8, 
    limit: SearchLimit, 
    hs: &[u64], 
    tt: &Arc<TranspositionTable>, 
    stop_flag: Arc<AtomicBool>,
    is_pondering: Arc<AtomicBool>,
    search_constants: SearchConstants 
) -> Option<Move> {
    
    TT_AGE.fetch_add(1, Ordering::Relaxed);
    NODES.store(0, Ordering::Relaxed);

    let actual_max_depth = max_depth;

    let mut tm = TimeManager::new(limit.clone());
    tm.start();

    init_thread_pool();

    while ACTIVE_WORKERS.load(Ordering::SeqCst) > 0 {
        std::thread::yield_now();
    }

    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    let current_run_id = RUN_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;

    let task = SearchTask {
        bd: bd.clone(),
        max_depth: actual_max_depth,
        tm: tm.clone(),
        hs: Arc::from(hs), 
        tt: Arc::clone(tt),
        stop_flag: Arc::clone(&stop_flag),
        is_pondering: Arc::clone(&is_pondering),
        search_constants: search_constants.clone(),
        run_id: current_run_id,
    };

    {
        let mut lock = TASK_MUTEX.lock().unwrap();
        *lock = Some(task.clone());
    }
    TASK_CONDVAR.notify_all();

    let result = MAIN_THREAD_DATA.with(|td_cell| {
        let mut td = td_cell.borrow_mut();
        td.search_constants = search_constants.clone();
        td.clear_for_search(); 
        
        search_worker(
            bd, actual_max_depth, &mut tm, hs, tt, &stop_flag, &is_pondering, true, &mut td
        )
    });
    
    stop_flag.store(true, Ordering::Relaxed);
    
    result.map(|(m, _)| m)
}
