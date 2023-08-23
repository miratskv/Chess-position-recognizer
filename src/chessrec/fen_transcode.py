import os
import chessrec.constants as const
import numpy as np
import cv2


"""
Util. functions used for transitions 
between "gold label encode" - "fen " - "image of the position"
"""

def overlay_png_images(
        backgr: np.ndarray, 
        forgr: np.ndarray, 
        y_offset: int, 
        x_offset: int
    ) -> np.ndarray:
    """
    Put the forgr img over the backgr  img with given offsets from the edges
    """
    start_y, end_y = y_offset, y_offset + forgr.shape[0]
    start_x, end_x  = x_offset, x_offset + forgr.shape[1]
    alpha_forgr = forgr[:, :, 3] / 255.0
    alpha_backgr = 1.0 - alpha_forgr
    backgr_ = alpha_backgr[...,None] * backgr[start_y:end_y, start_x:end_x, :-1]
    forgr_ = alpha_forgr[...,None] * forgr[:, :, :-1]
    backgr[start_y:end_y, start_x:end_x, :-1] = backgr_ + forgr_
    return backgr

def place_piece_on_board(
        board: np.ndarray,
        piece: np.ndarray,
        file: int,
        rank: int
    ) -> np.ndarray:
    """
    Takes an image of a board and puts an image of a piece on the given position.
    """
    piece_w = board.shape[0] // const.BOARD_FILES
    piece_h =  board.shape[1] // const.BOARD_RANKS
    piece = cv2.resize(piece, (piece_w, piece_h))
    x_offset = file*piece_w
    y_offset = (const.BOARD_RANKS - rank - 1)*piece_h
    return overlay_png_images(board, piece, y_offset, x_offset)

def decode_position(encoded_pos: np.ndarray) -> np.ndarray:
    """
    Takes an encoded position and recreates the image of the board
    """
    types = list(const.PIECES_ENCODING.keys())
    codes = list(const.PIECES_ENCODING.values())
    encoded_pos = np.flip(encoded_pos, axis = 0)
    position_image = cv2.imread(
        os.path.join(const.DEFAULT_ASSETS_FILE, const.DEFAULT_BOARD),
        cv2.IMREAD_UNCHANGED
    )

    position_image = cv2.cvtColor(position_image, cv2.COLOR_BGR2RGBA)
    position_image = cv2.resize(
        position_image, (const.DECODED_W, const.DECODED_H)
    )
    for rank in range(const.BOARD_RANKS):
        for file in range(const.BOARD_FILES):
          piece_encode = encoded_pos[rank, file]
          if piece_encode !=0 :
            piece = types[codes.index(piece_encode)]
            piece_image = cv2.imread(
                os.path.join(
                    const.DEFAULT_ASSETS_FILE, const.DEFAULT_SET_PIECE, f'{piece}.png'
                ),
                cv2.IMREAD_UNCHANGED
            )
            position_image = place_piece_on_board(
                position_image, 
                piece_image, 
                file, 
                rank
            )
        position_image = cv2.resize(
            position_image, (const.DECODED_W, const.DECODED_H)
        )
    return position_image

def encoding_to_FEN(encoded_pos: np.ndarray) -> str:
    """
    Converts cathegorical encoding to FEN notation.
    """
    FEN = ""
    types = list(const.PIECES_ENCODING_FEN.keys())
    codes = list(const.PIECES_ENCODING_FEN.values())
    current_empty = 0
    for row in encoded_pos:
        for piece_encode in row:
            if piece_encode == 0:
              current_empty +=1
            else:
              piece = types[codes.index(piece_encode)]
              if current_empty !=0:
                FEN += str(current_empty)
                current_empty = 0
              FEN += piece
        if current_empty !=0:
            FEN += str(current_empty)
            current_empty = 0
        FEN += "/"
    return FEN[:-1]
