// src/weigh.rs

use cozy_chess::{BitBoard, Board, Color, File, Move, Piece, Rank, Square};
use crate::search::{MATE_SCORE, ThreadData, SearchConstants};
use std::sync::atomic::{AtomicI32, Ordering};

// Precomputed King Center Activity Bonus (0x0 Floating-point Overhead)
const KING_CENTER_BONUS: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 5, 10, 15, 15, 10, 5, 0],
    [0, 5, 10, 15, 15, 10, 5, 0],
    [0, 0, 5, 10, 10, 5, 0, 0],
    [0, 0, 0, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];

const MG_VAL: [i32; 6] = [100, 330, 340, 500, 900, 0];
const EG_VAL: [i32; 6] = [120, 300, 350, 530, 925, 0];
const PHASE_W: [i32; 6] = [0, 1, 1, 2, 4, 0];
pub const SEE_VALS: [i32; 6] = [100, 300, 300, 500, 900, 20000];

// ============================================================================
//  DYNAMIC AND SYMMETRIC TUNABLE PST ARRAYS (32 parameters per table)
// ============================================================================

#[inline(always)]
pub fn get_sym_index(sq: usize) -> usize {
    let file = sq & 7;
    let rank = sq >> 3;
    let sym_file = if file > 3 { 7 - file } else { file };
    (rank << 2) + sym_file
}

// 1. Pawns (Middle Game) - Symmetric 32-values
pub static TUNABLE_PAWN_MG: [AtomicI32; 32] = [
    AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(0),
    AtomicI32::new(5),  AtomicI32::new(10), AtomicI32::new(10), AtomicI32::new(-20),
    AtomicI32::new(5),  AtomicI32::new(-5), AtomicI32::new(-10),AtomicI32::new(0),
    AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(20),
    AtomicI32::new(5),  AtomicI32::new(5),  AtomicI32::new(10), AtomicI32::new(25),
    AtomicI32::new(10), AtomicI32::new(10), AtomicI32::new(20), AtomicI32::new(30),
    AtomicI32::new(50), AtomicI32::new(50), AtomicI32::new(50), AtomicI32::new(50),
    AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(0),  AtomicI32::new(0),
];

// 2. Pawns (End Game) - Symmetric 32-values
pub static TUNABLE_PAWN_EG: [AtomicI32; 32] = [
    AtomicI32::new(0),   AtomicI32::new(0),   AtomicI32::new(0),   AtomicI32::new(0),
    AtomicI32::new(-10), AtomicI32::new(-10), AtomicI32::new(-10), AtomicI32::new(-10),
    AtomicI32::new(-5),  AtomicI32::new(-5),  AtomicI32::new(-5),  AtomicI32::new(-5),
    AtomicI32::new(5),   AtomicI32::new(5),   AtomicI32::new(5),   AtomicI32::new(15),
    AtomicI32::new(15),  AtomicI32::new(15),  AtomicI32::new(15),  AtomicI32::new(25),
    AtomicI32::new(30),  AtomicI32::new(30),  AtomicI32::new(30),  AtomicI32::new(40),
    AtomicI32::new(70),  AtomicI32::new(70),  AtomicI32::new(70),  AtomicI32::new(70),
    AtomicI32::new(0),   AtomicI32::new(0),   AtomicI32::new(0),   AtomicI32::new(0),
];

// 3. Knights - Symmetric 32-values
pub static TUNABLE_KNIGHT_PST: [AtomicI32; 32] = [
    AtomicI32::new(-50), AtomicI32::new(-40), AtomicI32::new(-30), AtomicI32::new(-30),
    AtomicI32::new(-40), AtomicI32::new(-20), AtomicI32::new(0),   AtomicI32::new(5),
    AtomicI32::new(-30), AtomicI32::new(5),   AtomicI32::new(10),  AtomicI32::new(15),
    AtomicI32::new(-30), AtomicI32::new(0),   AtomicI32::new(15),  AtomicI32::new(20),
    AtomicI32::new(-30), AtomicI32::new(5),   AtomicI32::new(15),  AtomicI32::new(20),
    AtomicI32::new(-30), AtomicI32::new(0),   AtomicI32::new(10),  AtomicI32::new(15),
    AtomicI32::new(-40), AtomicI32::new(-20), AtomicI32::new(0),   AtomicI32::new(0),
    AtomicI32::new(-50), AtomicI32::new(-40), AtomicI32::new(-30), AtomicI32::new(-30),
];

// static PSTs for pieces not actively being tuned using SPSA (or handled as fallback)
const BISHOP_PST: [i32; 64] = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -10,  5,  5, 15, 15,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
];

const ROOK_PST: [i32; 64] = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    25, 25, 25, 25, 25, 25, 25, 25,
     0,  0,  0,  0,  0,  0,  0,  0
];

const QUEEN_PST: [i32; 64] = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
];

const KING_MG: [i32; 64] = [
    20, 30, 10,-20,-20, 10, 30, 20,
    20, 20,-10,-30,-30,-10, 20, 20,
   -10,-20,-20,-40,-40,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
];

const KING_EG: [i32; 64] = [
   -50,-30,-30,-30,-30,-30,-30,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
];

#[inline(always)]
pub fn get_pst_val(p_idx: usize, sq_idx: usize, mg: bool) -> i32 {
    let base_val = if mg { MG_VAL[p_idx] } else { EG_VAL[p_idx] };
    let pst_offset = match p_idx {
        0 => { // Pawn
            let sym_idx = get_sym_index(sq_idx);
            if mg {
                TUNABLE_PAWN_MG[sym_idx].load(Ordering::Relaxed)
            } else {
                TUNABLE_PAWN_EG[sym_idx].load(Ordering::Relaxed)
            }
        }
        1 => { // Knight
            let sym_idx = get_sym_index(sq_idx);
            TUNABLE_KNIGHT_PST[sym_idx].load(Ordering::Relaxed)
        }
        2 => BISHOP_PST[sq_idx],
        3 => ROOK_PST[sq_idx],
        4 => QUEEN_PST[sq_idx],
        5 => if mg { KING_MG[sq_idx] } else { KING_EG[sq_idx] },
        _ => 0,
    };
    base_val + pst_offset
}

fn calc_base_eval(board: &Board) -> (i32, i32, i32) {
    let mut mg = 0; let mut eg = 0; let mut phase = 0;
    for p in Piece::ALL {
        let p_idx = p as usize;
        for sq in board.pieces(p) & board.colors(Color::White) {
            phase += PHASE_W[p_idx];
            mg += get_pst_val(p_idx, sq as usize, true);
            eg += get_pst_val(p_idx, sq as usize, false);
        }
        for sq in board.pieces(p) & board.colors(Color::Black) {
            phase += PHASE_W[p_idx];
            mg -= get_pst_val(p_idx, sq.flip_rank() as usize, true);
            eg -= get_pst_val(p_idx, sq.flip_rank() as usize, false);
        }
    }
    (mg, eg, phase)
}

// ============================================================================
//  DUAL BOARD WRAPPER
// ============================================================================
#[derive(Clone)]
pub struct SearchBoard {
    pub cozy: Board,
    pub base_mg: i32, 
    pub base_eg: i32, 
    pub phase: i32,   
}

impl SearchBoard {
    pub fn new(cozy: Board) -> Self {
        let (base_mg, base_eg, phase) = calc_base_eval(&cozy);
        Self { cozy, base_mg, base_eg, phase }
    }

    #[inline(always)]
    fn remove_piece(&mut self, piece: Piece, color: Color, sq: Square) {
        let p_idx = piece as usize;
        self.phase -= PHASE_W[p_idx];
        if color == Color::White {
            self.base_mg -= get_pst_val(p_idx, sq as usize, true);
            self.base_eg -= get_pst_val(p_idx, sq as usize, false);
        } else {
            self.base_mg += get_pst_val(p_idx, sq.flip_rank() as usize, true);
            self.base_eg += get_pst_val(p_idx, sq.flip_rank() as usize, false);
        }
    }

    #[inline(always)]
    fn add_piece(&mut self, piece: Piece, color: Color, sq: Square) {
        let p_idx = piece as usize;
        self.phase += PHASE_W[p_idx];
        if color == Color::White {
            self.base_mg += get_pst_val(p_idx, sq as usize, true);
            self.base_eg += get_pst_val(p_idx, sq as usize, false);
        } else {
            self.base_mg -= get_pst_val(p_idx, sq.flip_rank() as usize, true);
            self.base_eg -= get_pst_val(p_idx, sq.flip_rank() as usize, false);
        }
    }

    #[inline(always)]
    pub fn play(&mut self, m: Move) {
        let color = self.cozy.side_to_move();
        let piece = self.cozy.piece_on(m.from).unwrap_or(Piece::Pawn);
        let captured = self.cozy.piece_on(m.to);
        let is_castling = piece == Piece::King && captured == Some(Piece::Rook) && self.cozy.color_on(m.to) == Some(color);

        if is_castling {
            self.remove_piece(Piece::King, color, m.from);
            self.remove_piece(Piece::Rook, color, m.to);
            let is_kingside = m.to.file() > m.from.file();
            let king_to_file = if is_kingside { File::G } else { File::C };
            let rook_to_file = if is_kingside { File::F } else { File::D };
            let rank = m.from.rank();
            self.add_piece(Piece::King, color, Square::new(king_to_file, rank));
            self.add_piece(Piece::Rook, color, Square::new(rook_to_file, rank));
        } else {
            self.remove_piece(piece, color, m.from);
            let mut mut_cap_p = captured;
            let mut cap_sq = m.to;
            if piece == Piece::Pawn && m.to.file() != m.from.file() && captured.is_none() {
                mut_cap_p = Some(Piece::Pawn);
                cap_sq = Square::new(m.to.file(), m.from.rank());
            }
            if let Some(p) = mut_cap_p { self.remove_piece(p, !color, cap_sq); }
            let placed_piece = m.promotion.unwrap_or(piece);
            self.add_piece(placed_piece, color, m.to);
        }

        self.cozy.play(m);
    }
    
    pub fn null_move(&self) -> Option<Self> {
        if let Some(new_cozy) = self.cozy.null_move() {
            Some(Self { cozy: new_cozy, base_mg: self.base_mg, base_eg: self.base_eg, phase: self.phase })
        } else { None }
    }
}

// ============================================================================
//  THREAD LOCAL CACHES & DATA
// ============================================================================
const LOCAL_EVAL_CACHE_SIZE: usize = 131072; 
#[derive(Clone, Copy)]
struct LocalEvalEntry { key: u64, score: i32, valid: bool }

pub struct LocalEvalCache { table: Box<[LocalEvalEntry]> }
impl LocalEvalCache {
    pub fn new() -> Self { Self { table: vec![LocalEvalEntry { key: 0, score: 0, valid: false }; LOCAL_EVAL_CACHE_SIZE].into_boxed_slice() } }
    #[inline(always)] pub fn probe(&self, key: u64) -> Option<i32> {
        let idx = (key as usize) & (LOCAL_EVAL_CACHE_SIZE - 1);
        let entry = &self.table[idx];
        if entry.valid && entry.key == key { Some(entry.score) } else { None }
    }
    #[inline(always)] pub fn store(&mut self, key: u64, score: i32) {
        let idx = (key as usize) & (LOCAL_EVAL_CACHE_SIZE - 1);
        self.table[idx] = LocalEvalEntry { key, score, valid: true };
    }
}

const PAWN_CACHE_SIZE: usize = 131072;
#[derive(Clone, Copy)]
struct PawnCacheEntry { key: u64, score: i32 }
pub struct PawnCache { table: Box<[PawnCacheEntry]> }
impl PawnCache {
    pub fn new() -> Self { Self { table: vec![PawnCacheEntry { key: 0, score: 0 }; PAWN_CACHE_SIZE].into_boxed_slice() } }
    pub fn probe(&self, key: u64) -> Option<i32> {
        let idx = (key as usize) & (PAWN_CACHE_SIZE - 1);
        if self.table[idx].key == key { Some(self.table[idx].score) } else { None }
    }
    pub fn store(&mut self, key: u64, score: i32) {
        let idx = (key as usize) & (PAWN_CACHE_SIZE - 1);
        self.table[idx] = PawnCacheEntry { key, score };
    }
}

// ============================================================================
//  BITBOARD HELPERS & EVALUATION
// ============================================================================
const FILE_A_BB: u64 = 0x0101010101010101;
const FILE_H_BB: u64 = 0x8080808080808080;

#[inline(always)] fn file_bb(f: File) -> BitBoard { BitBoard(FILE_A_BB << (f as usize)) }
#[inline(always)] fn adj_files_bb(f: File) -> BitBoard {
    let f_bb = FILE_A_BB << (f as usize);
    BitBoard(((f_bb >> 1) & !FILE_H_BB) | ((f_bb << 1) & !FILE_A_BB))
}
#[inline(always)] fn forward_bb(color: Color, r: Rank) -> BitBoard {
    BitBoard(match color {
        Color::White => !0u64 << ((r as usize + 1) * 8),
        Color::Black => !0u64 >> ((8 - r as usize) * 8),
    })
}

#[inline(always)]
fn pawn_attacks(pawns: BitBoard, color: Color) -> BitBoard {
    if color == Color::White { BitBoard(((pawns.0 & !FILE_A_BB) << 7) | ((pawns.0 & !FILE_H_BB) << 9)) } 
    else { BitBoard(((pawns.0 & !FILE_H_BB) >> 7) | ((pawns.0 & !FILE_A_BB) >> 9)) }
}

#[inline(always)] fn taper(mg: i32, eg: i32, phase_mg: i32, phase_eg: i32) -> i32 { (mg * phase_mg + eg * phase_eg + 12) / 24 }

fn find_passed_pawns(board: &Board, color: Color) -> BitBoard {
    let mut passed = BitBoard(0);
    let us = board.colors(color);
    let them = board.colors(!color);
    let pawns = board.pieces(Piece::Pawn) & us;
    let enemy_pawns = board.pieces(Piece::Pawn) & them;
    
    for sq in pawns {
        let f = sq.file();
        let r = sq.rank();
        let file_mask = file_bb(f);
        let adj_mask = adj_files_bb(f);
        let forward_mask = forward_bb(color, r);
        let passed_zone = (file_mask | adj_mask) & forward_mask;
        if (passed_zone & enemy_pawns).is_empty() {
            passed.0 |= 1u64 << (sq as usize);
        }
    }
    passed
}

fn mating_endgame_bonus(board: &Board) -> i32 {
    let w_pawns = (board.pieces(Piece::Pawn) & board.colors(Color::White)).len();
    let b_pawns = (board.pieces(Piece::Pawn) & board.colors(Color::Black)).len();
    let w_knights = (board.pieces(Piece::Knight) & board.colors(Color::White)).len();
    let b_knights = (board.pieces(Piece::Knight) & board.colors(Color::Black)).len();
    let w_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::White)).len();
    let b_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::Black)).len();
    let w_rooks = (board.pieces(Piece::Rook) & board.colors(Color::White)).len();
    let b_rooks = (board.pieces(Piece::Rook) & board.colors(Color::Black)).len();
    let w_queens = (board.pieces(Piece::Queen) & board.colors(Color::White)).len();
    let b_queens = (board.pieces(Piece::Queen) & board.colors(Color::Black)).len();

    let w_total = w_pawns + w_knights + w_bishops + w_rooks + w_queens;
    let b_total = b_pawns + b_knights + b_bishops + b_rooks + b_queens;

    let mut bonus = 0;

    if b_total == 0 && w_total > 0 {
        if w_queens == 1 && w_rooks == 0 && w_bishops == 0 && w_knights == 0 && w_pawns == 0 {
            bonus += 300; // KQ vs K
        } else if w_rooks == 1 && w_queens == 0 && w_bishops == 0 && w_knights == 0 && w_pawns == 0 {
            bonus += 150; // KR vs K
        } else if w_bishops == 1 && w_knights == 1 && w_queens == 0 && w_rooks == 0 && w_pawns == 0 {
            bonus += 100; // KBN vs K
        }
    }

    if w_total == 0 && b_total > 0 {
        if b_queens == 1 && b_rooks == 0 && b_bishops == 0 && b_knights == 0 && b_pawns == 0 {
            bonus -= 300; // KQ vs K
        } else if b_rooks == 1 && b_queens == 0 && b_bishops == 0 && b_knights == 0 && b_pawns == 0 {
            bonus -= 150; // KR vs K
        } else if b_bishops == 1 && b_knights == 1 && b_queens == 0 && b_rooks == 0 && b_pawns == 0 {
            bonus -= 100; // KBN vs K
        }
    }

    bonus
}

#[inline(always)]
fn endgame_scale_factor(board: &Board, score: i32) -> i32 {
    let w_pawns = (board.pieces(Piece::Pawn) & board.colors(Color::White)).len() as i32;
    let b_pawns = (board.pieces(Piece::Pawn) & board.colors(Color::Black)).len() as i32;
    let w_knights = (board.pieces(Piece::Knight) & board.colors(Color::White)).len() as i32;
    let b_knights = (board.pieces(Piece::Knight) & board.colors(Color::Black)).len() as i32;
    let w_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::White)).len() as i32;
    let b_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::Black)).len() as i32;
    let w_rooks = (board.pieces(Piece::Rook) & board.colors(Color::White)).len() as i32;
    let b_rooks = (board.pieces(Piece::Rook) & board.colors(Color::Black)).len() as i32;
    let w_queens = (board.pieces(Piece::Queen) & board.colors(Color::White)).len() as i32;
    let b_queens = (board.pieces(Piece::Queen) & board.colors(Color::Black)).len() as i32;

    let w_non_pawns = w_knights + w_bishops + w_rooks + w_queens;
    let b_non_pawns = b_knights + b_bishops + b_rooks + b_queens;

    let mut scale = 256;

    if w_pawns == 1 && b_pawns == 0 && w_non_pawns == 1 && w_bishops == 1 && b_non_pawns == 0 && score > 0 {
        let p_sq = (board.pieces(Piece::Pawn) & board.colors(Color::White)).into_iter().next().unwrap();
        let b_sq = (board.pieces(Piece::Bishop) & board.colors(Color::White)).into_iter().next().unwrap();
        let f = p_sq.file();
        if f == File::A || f == File::H {
            let promo_sq_file = if f == File::A { 0 } else { 7 };
            let promo_color = (7 + promo_sq_file) % 2; 
            let bishop_color = (b_sq.rank() as usize + b_sq.file() as usize) % 2;
            if promo_color != bishop_color {
                let b_king_sq = (board.pieces(Piece::King) & board.colors(Color::Black)).into_iter().next().unwrap();
                let dist_to_corner = (b_king_sq.file() as usize).abs_diff(promo_sq_file) + (7 - b_king_sq.rank() as usize);
                if dist_to_corner <= 2 {
                    return 0; 
                }
            }
        }
    }

    if b_pawns == 1 && w_pawns == 0 && b_non_pawns == 1 && b_bishops == 1 && w_non_pawns == 0 && score < 0 {
        let p_sq = (board.pieces(Piece::Pawn) & board.colors(Color::Black)).into_iter().next().unwrap();
        let b_sq = (board.pieces(Piece::Bishop) & board.colors(Color::Black)).into_iter().next().unwrap();
        let f = p_sq.file();
        if f == File::A || f == File::H {
            let promo_sq_file = if f == File::A { 0 } else { 7 };
            let promo_color = (0 + promo_sq_file) % 2; 
            let bishop_color = (b_sq.rank() as usize + b_sq.file() as usize) % 2;
            if promo_color != bishop_color {
                let w_king_sq = (board.pieces(Piece::King) & board.colors(Color::White)).into_iter().next().unwrap();
                let dist_to_corner = (w_king_sq.file() as usize).abs_diff(promo_sq_file) + w_king_sq.rank() as usize;
                if dist_to_corner <= 2 {
                    return 0; 
                }
            }
        }
    }

    if w_pawns == 0 && score > 0 {
        if w_non_pawns == 0 { return 0; }
        if w_non_pawns == 1 && (w_knights == 1 || w_bishops == 1) { return 0; }
        if w_rooks == 1 && w_non_pawns == 1 && b_non_pawns == 1 && (b_knights == 1 || b_bishops == 1) { return 32; } 
        if w_rooks == 1 && w_non_pawns == 1 && b_rooks == 1 && b_non_pawns == 1 { return 32; }
        scale = 100;
    }

    if b_pawns == 0 && score < 0 {
        if b_non_pawns == 0 { return 0; }
        if b_non_pawns == 1 && (b_knights == 1 || b_bishops == 1) { return 0; }
        if b_rooks == 1 && b_non_pawns == 1 && w_non_pawns == 1 && (w_knights == 1 || w_bishops == 1) { return 32; } 
        if b_rooks == 1 && b_non_pawns == 1 && w_rooks == 1 && w_non_pawns == 1 { return 32; }
        scale = 100;
    }

    if w_non_pawns == 1 && b_non_pawns == 1 && w_bishops == 1 && b_bishops == 1 {
        let w_sq = (board.pieces(Piece::Bishop) & board.colors(Color::White)).into_iter().next().unwrap();
        let b_sq = (board.pieces(Piece::Bishop) & board.colors(Color::Black)).into_iter().next().unwrap();
        if (w_sq.rank() as usize + w_sq.file() as usize) % 2 != (b_sq.rank() as usize + b_sq.file() as usize) % 2 {
            if w_pawns + b_pawns <= 4 {
                scale = 64; 
            } else {
                scale = 128; 
            }
        }
    }
    scale
}

#[inline(always)]
pub fn evaluate(bd: &SearchBoard, hash: u64, td: &mut ThreadData) -> i32 {
    if let Some(cached) = td.eval_cache.probe(hash) { return cached; }

    let board = &bd.cozy;
    let mg_score = bd.base_mg; let eg_score = bd.base_eg; let phase = bd.phase;
    let phase_mg = phase.min(24); let phase_eg = 24 - phase_mg;
    let mut score = taper(mg_score, eg_score, phase_mg, phase_eg);

    let w_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::White)).len();
    let b_bishops = (board.pieces(Piece::Bishop) & board.colors(Color::Black)).len();
    if w_bishops >= 2 { score += taper(40, 30, phase_mg, phase_eg); }
    if b_bishops >= 2 { score -= taper(40, 30, phase_mg, phase_eg); }

    let total_pawns = board.pieces(Piece::Pawn).len() as i32;
    let w_knights = (board.pieces(Piece::Knight) & board.colors(Color::White)).len() as i32;
    let b_knights = (board.pieces(Piece::Knight) & board.colors(Color::Black)).len() as i32;
    if total_pawns < 8 {
        let penalty = (8 - total_pawns) * 4;
        score -= w_knights * penalty;
        score += b_knights * penalty;
    }

    if phase_mg > 16 {
        let w_knights_home = (board.pieces(Piece::Knight) & board.colors(Color::White) & BitBoard(0x0000000000000042)).len() as i32;
        let b_knights_home = (board.pieces(Piece::Knight) & board.colors(Color::Black) & BitBoard(0x4200000000000000)).len() as i32;
        let w_bishops_home = (board.pieces(Piece::Bishop) & board.colors(Color::White) & BitBoard(0x0000000000000024)).len() as i32;
        let b_bishops_home = (board.pieces(Piece::Bishop) & board.colors(Color::Black) & BitBoard(0x2400000000000000)).len() as i32;
        score -= (w_knights_home + w_bishops_home) * 15;
        score += (b_knights_home + b_bishops_home) * 15;
    }

    score += pawn_eval_both(board, &mut td.pawn_cache);

    let w_passed = find_passed_pawns(board, Color::White);
    let b_passed = find_passed_pawns(board, Color::Black);

    let (w_mob, w_atk_count, w_atk_weight) = piece_eval(board, Color::White, w_passed, b_passed);
    let (b_mob, b_atk_count, b_atk_weight) = piece_eval(board, Color::Black, b_passed, w_passed);
    score += w_mob - b_mob;
    score -= calculate_king_penalty(board, Color::White, b_atk_count, b_atk_weight, phase_mg);
    score += calculate_king_penalty(board, Color::Black, w_atk_count, w_atk_weight, phase_mg);

    score += mating_endgame_bonus(board);

    let scale_factor = endgame_scale_factor(board, score);
    if score.abs() < MATE_SCORE - 1000 {
        score = (score * scale_factor) / 256;
    }
    let mut final_score = if board.side_to_move() == Color::White { score } else { -score };
    final_score += td.search_constants.tempo_bonus;

    td.eval_cache.store(hash, final_score);
    final_score
}

#[inline(always)]
fn pawn_eval_both(board: &Board, pawn_cache: &mut PawnCache) -> i32 {
    let w_pawns = board.pieces(Piece::Pawn) & board.colors(Color::White);
    let b_pawns = board.pieces(Piece::Pawn) & board.colors(Color::Black);
    let key = w_pawns.0.wrapping_mul(0x9E3779B97F4A7C15) ^ b_pawns.0.wrapping_mul(0xC6A4A7935BD1E995);
    if let Some(score) = pawn_cache.probe(key) { return score; }
    let score = pawn_eval_fast(board, Color::White) - pawn_eval_fast(board, Color::Black);
    pawn_cache.store(key, score);
    score
}

#[inline(always)]
fn pawn_eval_fast(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let us = board.colors(color);
    let them = board.colors(!color);
    let pawns = board.pieces(Piece::Pawn) & us;
    let enemy_pawns = board.pieces(Piece::Pawn) & them;

    let our_pawn_attacks = pawn_attacks(pawns, color);
    let enemy_pawn_attacks = pawn_attacks(enemy_pawns, !color);

    for sq in pawns {
        let f = sq.file(); let r = sq.rank();
        let file_mask = file_bb(f); let adj_mask = adj_files_bb(f); let forward_mask = forward_bb(color, r);

        let on_file = (pawns & file_mask).len() as i32;
        if on_file > 1 { score -= (on_file - 1) * 10; }
        if (pawns & adj_mask).is_empty() { score -= 15; }

        let neighbor_mask = cozy_chess::get_king_moves(sq) & adj_files_bb(f);
        let connected_count = (pawns & neighbor_mask).len() as i32;
        if connected_count > 0 { score += (connected_count * 8) / 2; }

        let passed_zone = (file_mask | adj_mask) & forward_mask;
        if (passed_zone & enemy_pawns).is_empty() {
            let rel_rank = if color == Color::White { r as i32 } else { 7 - r as i32 };
            let mut passer_bonus = 20 + rel_rank * 25; 
            
            let support_zone = cozy_chess::get_pawn_attacks(sq, !color);
            if !(pawns & support_zone).is_empty() { passer_bonus += 15 + rel_rank * 10; }
            score += passer_bonus;
        }

        let ahead = BitBoard(if color == Color::White { 1u64 << (sq as usize + 8) } else { 1u64 << (sq as usize - 8) });
        let can_be_supported = !(our_pawn_attacks & ahead).is_empty();
        let is_held_by_enemy = !(enemy_pawn_attacks & ahead).is_empty();
        if !can_be_supported && is_held_by_enemy { score -= 12; }
    }
    score
}

#[inline(always)]
fn piece_eval(board: &Board, color: Color, our_passed: BitBoard, enemy_passed: BitBoard) -> (i32, i32, i32) {
    let mut mobility = 0; let mut attacker_count = 0; let mut attack_weight = 0;
    let occ = board.occupied(); let us = board.colors(color);
    
    let king_bb = board.pieces(Piece::King) & board.colors(!color);
    if king_bb.is_empty() { return (0, 0, 0); }
    let enemy_king_sq = king_bb.into_iter().next().unwrap();
    
    let enemy_king_ring = cozy_chess::get_king_moves(enemy_king_sq) | BitBoard(1u64 << enemy_king_sq as usize);
    let all_pawns = board.pieces(Piece::Pawn); let us_pawns = all_pawns & us; let them_pawns = all_pawns & board.colors(!color);
    let enemy_pawn_attacks = pawn_attacks(them_pawns, !color);

    for sq in board.pieces(Piece::Knight) & us {
        let attacks = cozy_chess::get_knight_moves(sq);
        let safe_mob = attacks & !us & !enemy_pawn_attacks;
        mobility += safe_mob.len() as i32 * 3;
        let rel_rank = if color == Color::White { sq.rank() as i32 } else { 7 - sq.rank() as i32 };
        if rel_rank >= 3 && rel_rank <= 5 {
            let outpost_zone = forward_bb(color, sq.rank()) & adj_files_bb(sq.file());
            if (them_pawns & outpost_zone).is_empty() && !(us_pawns & cozy_chess::get_pawn_attacks(sq, !color)).is_empty() { mobility += 15; }
        }
        let ring = (attacks & enemy_king_ring).len() as i32;
        if ring > 0 { attacker_count += 1; attack_weight += ring * 2; }
    }

    for sq in board.pieces(Piece::Bishop) & us {
        let attacks = cozy_chess::get_bishop_moves(sq, occ);
        let safe_mob = attacks & !us & !enemy_pawn_attacks;
        mobility += safe_mob.len() as i32 * 3;

        let is_light_sq = (sq.rank() as usize + sq.file() as usize) % 2 == 1;
        let same_color_mask = if is_light_sq { 0xAA55AA55AA55AA55 } else { 0x55AA55AA55AA55AA };
        let bad_pawns_count = (board.pieces(Piece::Pawn) & us & BitBoard(same_color_mask)).len() as i32;
        mobility -= bad_pawns_count * 2; 

        let ring = (attacks & enemy_king_ring).len() as i32;
        if ring > 0 { attacker_count += 1; attack_weight += ring * 2; }
    }

    for sq in board.pieces(Piece::Rook) & us {
        let attacks = cozy_chess::get_rook_moves(sq, occ);
        let safe_mob = attacks & !us & !enemy_pawn_attacks;
        mobility += safe_mob.len() as i32 * 2;
        let f_mask = file_bb(sq.file());
        if (f_mask & all_pawns).is_empty() { mobility += 20; }
        else if (f_mask & us_pawns).is_empty() { mobility += 10; }
        let rel_rank = if color == Color::White { sq.rank() } else { sq.rank().flip() };
        if rel_rank == Rank::Seventh && !(them_pawns & BitBoard(if color == Color::White { 0x00FF000000000000 } else { 0x000000000000FF00 })).is_empty() { mobility += 25; }

        let other_rooks = (board.pieces(Piece::Rook) & us) & !BitBoard(1u64 << sq as usize);
        if !(other_rooks & (f_mask | BitBoard(0xFFu64 << (sq.rank() as usize * 8)))).is_empty() {
            mobility += 15;
        }

        let our_passed_on_file = our_passed & f_mask;
        if !our_passed_on_file.is_empty() {
            let pawn_sq = our_passed_on_file.into_iter().next().unwrap();
            let is_behind = if color == Color::White { sq.rank() < pawn_sq.rank() } else { sq.rank() > pawn_sq.rank() };
            if is_behind {
                mobility += 20; 
            }
        }
        let enemy_passed_on_file = enemy_passed & f_mask;
        if !enemy_passed_on_file.is_empty() {
            let pawn_sq = enemy_passed_on_file.into_iter().next().unwrap();
            let is_behind = if color == Color::White { sq.rank() < pawn_sq.rank() } else { sq.rank() > pawn_sq.rank() };
            if is_behind {
                mobility += 15; 
            }
        }

        let ring = (attacks & enemy_king_ring).len() as i32;
        if ring > 0 { attacker_count += 1; attack_weight += ring * 3; }
    }

    for sq in board.pieces(Piece::Queen) & us {
        let attacks = cozy_chess::get_bishop_moves(sq, occ) | cozy_chess::get_rook_moves(sq, occ);
        let safe_mob = attacks & !us & !enemy_pawn_attacks;
        mobility += safe_mob.len() as i32 * 2;
        let minor_home_mask = if color == Color::White { 0x0000000000000066 } else { 0x6600000000000000 };
        let minors_at_home = (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)) & us & BitBoard(minor_home_mask);
        if !minors_at_home.is_empty() && (sq.rank() != Rank::First && sq.rank() != Rank::Eighth) { mobility -= 10 * minors_at_home.len() as i32; }
        let ring = (attacks & enemy_king_ring).len() as i32;
        if ring > 0 { attacker_count += 1; attack_weight += ring * 5; }
    }

    (mobility, attacker_count, attack_weight)
}

#[inline(always)]
fn calculate_king_penalty(board: &Board, color: Color, mut attacker_count: i32, mut attack_weight: i32, phase: i32) -> i32 {
    let king_bb = board.pieces(Piece::King) & board.colors(color);
    if king_bb.is_empty() { return 0; }
    let king_sq = king_bb.into_iter().next().unwrap();
    
    let king_ring = cozy_chess::get_king_moves(king_sq);
    let us = board.colors(color); let them = board.colors(!color);
    let pawns = board.pieces(Piece::Pawn); let us_pawns = pawns & us; let them_pawns = pawns & them;
    let mut penalty = 0;

    let k_file = king_sq.file() as usize; let k_rank = king_sq.rank() as usize;
    let is_king_home = if color == Color::White { k_rank <= 1 } else { k_rank >= 6 };

    let has_short = board.castle_rights(color).short.is_some();
    let has_long = board.castle_rights(color).long.is_some();
    if is_king_home && king_sq.file() == File::E {
        if !has_short && !has_long {
            penalty += 30; 
        }
    }

    let mut missing_shield = 0; let mut shield_weakness = 0;
    if is_king_home && phase > 8 {
        let min_f = k_file.saturating_sub(1); let max_f = (k_file + 1).min(7);
        for f in min_f..=max_f {
            let f_mask = file_bb(File::index(f)); let our_pawns_on_file = f_mask & us_pawns;
            if our_pawns_on_file.is_empty() { missing_shield += 1; shield_weakness += 40; } 
            else {
                let pawn_sq = if color == Color::White { our_pawns_on_file.into_iter().next().unwrap() } else { our_pawns_on_file.into_iter().last().unwrap() };
                let p_rank = pawn_sq.rank() as usize;
                let steps = if color == Color::White { p_rank.saturating_sub(1) } else { 6_usize.saturating_sub(p_rank) };
                let is_f_or_c = f == 5 || f == 2;
                shield_weakness += match steps { 0 => 0, 1 => if is_f_or_c { 15 } else { 5 }, _ => 30 };
            }
        }
        penalty += (shield_weakness * phase) / 24;
    }

    if is_king_home {
        let min_f = k_file.saturating_sub(1); let max_f = (k_file + 1).min(7);
        let mut file_penalty = 0;
        for f in min_f..=max_f {
            let f_mask = file_bb(File::index(f));
            if (f_mask & us_pawns).is_empty() {
                if (f_mask & them_pawns).is_empty() { file_penalty += 25; } else { file_penalty += 15; }
            }
        }
        penalty += (file_penalty * phase) / 24;
    }

    let enemy_pieces = board.colors(!color) & !board.pieces(Piece::King);
    for sq in enemy_pieces {
        let dist = (k_file.abs_diff(sq.file() as usize) + k_rank.abs_diff(sq.rank() as usize)) as i32;
        if dist <= 3 { attack_weight += (4 - dist) * 2; attacker_count += 1; }
    }

    if phase < 8 {
        let activity_bonus = KING_CENTER_BONUS[k_rank][k_file];
        penalty -= activity_bonus;
    }

    let enemy_attack_power = (board.pieces(Piece::Queen) & them).len() * 4
        + (board.pieces(Piece::Rook) & them).len() * 2
        + ((board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)) & them).len();

    let enemy_has_queen = !(board.pieces(Piece::Queen) & them).is_empty();

    if enemy_attack_power < 2 { 
        if !enemy_has_queen {
            penalty = (penalty * 3) / 10; 
        }
        return penalty.max(0); 
    }

    let pawn_defenders = (king_ring & us_pawns).len() as i32;
    let minor_defenders = (king_ring & us & (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop))).len() as i32;
    let defense_power = (pawn_defenders * 5) + (minor_defenders * 3);

    let enemy_pawns_in_ring = (king_ring & them_pawns).len() as i32;
    if enemy_pawns_in_ring > 0 { attacker_count += 1; attack_weight += enemy_pawns_in_ring * 6; }

    if attacker_count >= 2 {
        if !enemy_has_queen { attack_weight /= 2; }
        if missing_shield > 0 { attack_weight += missing_shield * 8; }
        attack_weight = (attack_weight - defense_power).max(0);

        if attack_weight > 0 {
            const SAFETY_TABLE:[i32; 20] = [0, 0, 2, 5, 10, 18, 28, 40, 55, 72, 90, 110, 132, 155, 180, 205, 230, 255, 280, 300];
            let idx = (attack_weight as usize).min(19);
            let phase_factor = phase.max(4).min(24);
            penalty += (SAFETY_TABLE[idx] * phase_factor) / 24;
        }
    }

    if !enemy_has_queen {
        penalty = (penalty * 3) / 10; 
    }

    penalty.max(0)
}
