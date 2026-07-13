const std = @import("std");
const GameState = @import("../gamestate.zig").GameState;
const Piece = @import("../piece.zig").Piece;
const PieceKind = @import("../piece.zig").PieceKind;

pub const N_PIECE_TYPES: usize = 6;
pub const N_SQUARES: usize = 64;
pub const N_COLORS: usize = 2;
pub const N_KING_BUCKETS: usize = 16;
pub const INPUT_SIZE: usize = N_PIECE_TYPES * N_SQUARES * N_COLORS * N_KING_BUCKETS;

pub const KING_SQUARE_INDICES = [64]u8{
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  9,  10, 11, 11, 10, 9,  8,
    8,  9,  10, 11, 11, 10, 9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
};

pub const Feature = struct {
    index_white: usize,
    index_black: usize,
};

pub fn getKingSquareIndex(king_square: u8, king_color: i8) u8 {
    var sq = king_square ^ 56;

    if (king_color == 0) {
        sq ^= 56;
    }
    return KING_SQUARE_INDICES[sq];
}

pub fn getFeatureIndex(
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
    view: i8,
    king_square: u8,
) usize {
    const ks_index = getKingSquareIndex(king_square, view);

    var sq = square ^ 56;

    if (view == 0) {
        sq ^= 56;
    }

    if ((king_square & 0x4) != 0) {
        sq ^= 7;
    }

    const pt = @intFromEnum(piece_kind);
    const color_idx: usize = if (piece_color == view) 1 else 0;

    const index = @as(usize, sq) +
        @as(usize, pt) * 64 +
        color_idx * 384 +
        @as(usize, ks_index) * 768;

    return index;
}

test "Koivisto feature coordinates use A1 origin" {
    const feature = getFeatureWithKings(.Pawn, 1, 52, 60, 4);

    try std.testing.expectEqual(@as(usize, 2699), feature.index_white);
    try std.testing.expectEqual(@as(usize, 2355), feature.index_black);
}

pub fn getFeature(
    state: *const GameState,
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
) Feature {
    const white_king = findKing(state, 1) orelse return Feature{ .index_white = 0, .index_black = 0 };
    const black_king = findKing(state, 0) orelse return Feature{ .index_white = 0, .index_black = 0 };

    return Feature{
        .index_white = getFeatureIndex(piece_kind, piece_color, square, 1, white_king),
        .index_black = getFeatureIndex(piece_kind, piece_color, square, 0, black_king),
    };
}

pub fn getFeatureWithKings(
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
    white_king: u8,
    black_king: u8,
) Feature {
    return Feature{
        .index_white = getFeatureIndex(piece_kind, piece_color, square, 1, white_king),
        .index_black = getFeatureIndex(piece_kind, piece_color, square, 0, black_king),
    };
}

pub fn findKing(state: *const GameState, color: i8) ?u8 {
    for (state.board.squares, 0..) |square, idx| {
        if (square.piece) |piece| {
            if (piece.kind == .King and piece.color == color) {
                return @intCast(idx);
            }
        }
    }
    return null;
}

pub fn getActiveFeatures(state: *const GameState, allocator: std.mem.Allocator) ![]Feature {
    var features = std.ArrayList(Feature).init(allocator);

    for (state.board.squares, 0..) |square, idx| {
        if (square.piece) |piece| {
            const feature = getFeature(state, piece.kind, piece.color, @intCast(idx));
            try features.append(feature);
        }
    }

    return features.toOwnedSlice();
}
