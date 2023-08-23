import os

"""
File with some usefull constants and notations
"""

STOCKFISH_PATH = ""

DEFAULT_ASSETS_FILE = os.path.join(os.path.dirname(__file__),"assets","default")
DEFAULT_BOARD = "board_default.png"
DEFAULT_SET_PIECE = "pieces"

DECODED_H = 256
DECODED_W = 256

NUMBER_OF_PIECES: int = 32
BOARD_RANKS: int = 8
BOARD_FILES: int = 8
N_SQUARES: int = BOARD_FILES*BOARD_RANKS

PLAYERS: list[str] = ['w', 'b']
PIECE_TYPES: list[str] = ['K', 'Q', 'R', 'B', 'N', 'P']
PIECE_INIT_NUMBER: list[int] = [2, 2, 4, 4, 4, 16]
SQUARE_CLASSES = 2*len(PIECE_TYPES) + 1

PIECES_ENCODING: dict[str, int] = {
'wK': 1, 'wQ': 2, 'wR': 3, 'wB': 4, 'wN': 5,'wP': 6,
'bK': 7, 'bQ': 8, 'bR': 9, 'bB': 10, 'bN': 11,'bP': 12,
}

PIECES_ENCODING_FEN: dict[str, int] = {
'K': 1, 'Q': 2, 'R': 3, 'B': 4, 'N': 5,'P': 6,
'k': 7, 'q': 8, 'r': 9, 'b': 10, 'n': 11,'p': 12,
} 
