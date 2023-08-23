import os
import random
import numpy as np
import tensorflow as tf
import cv2

import chessrec.constants as const
import chessrec.fen_transcode as fen_transcode

class ChessBoardGenerator():
    """
    Generator of the images with random chess positions and random backgrounds.
    The tensorflow data generator can be obrained as self.tfGenerator()
    """
    BACKGROUND_H: int = 576
    BACKGROUND_W: int = 1024
    COL_CHANNELS: int = 3
    IMAGE_SUPP_FORMATS = [".jpg", ".png"]
    def __init__(self, 
              boards_imgs_path: str = '', 
              piece_sets_path: str = '',
              background_im_path: str = '', 
              out_board_H: int = 256, 
              out_board_W: int = 256,
              output_min_size: int = 96,
              output_max_size: int = 512) -> None:

        # Setting assets path
        self.piece_sets_path = piece_sets_path
        self.pieces_sets = [os.path.join(piece_sets_path, f) 
                                for f in os.listdir(piece_sets_path) 
                                  if os.path.isdir(os.path.join(piece_sets_path, f))]
        self.boards = [os.path.join(boards_imgs_path, img) 
                          for img in os.listdir(boards_imgs_path) 
                            if os.path.splitext(img)[-1].lower() in self.IMAGE_SUPP_FORMATS]
        self.backgrounds = [os.path.join(background_im_path, img) 
                               for img in os.listdir(background_im_path) 
                                  if os.path.splitext(img)[-1].lower() in self.IMAGE_SUPP_FORMATS]

        # formating sizes of the output images
        self.output_min_size = output_min_size
        self.output_max_size = output_max_size
        self.out_board_H = out_board_H
        self.out_board_W = out_board_W
        # Preparing all possible board coordinations
        coordinates = np.meshgrid(
                        np.arange(const.BOARD_RANKS), 
                        np.arange(const.BOARD_FILES))
        self._coordinates = np.array(coordinates).T.reshape([-1,2])
        # Switch whether to generate only boards or not
        self._only_boards = True

    def _pieces_probs(self, n_pieces_on_board: int) -> np.ndarray:
        """
        Generates sample probabilities for each type of piece. 
        To generate more realistic game positions, one could 
        account for the total number of pieces left on board, assert 
        two kings on board etc.
        """

        n_types = len(const.PIECE_INIT_NUMBER)
        piece_type_probs = np.ones(n_types)/n_types
        return piece_type_probs

    def _sample_pieces(self):
        """
        Sample a set of pieces to be placed on board from self._pieces_probs()
        """
        n_pieces_on_board = np.random.randint(
                              low = 1, 
                              high = const.NUMBER_OF_PIECES +1)
        piece_type_probs = self._pieces_probs(n_pieces_on_board)
        pieces = []
        for _ in range(n_pieces_on_board):
            player = np.random.choice(const.PLAYERS)
            piece = np.random.choice(
                        const.PIECE_TYPES, 
                        p=piece_type_probs)
            pieces.append(f'{player}{piece}')
        return pieces

    def _sample_positions(self, pieces: list[str]) -> np.ndarray:
        """
        Sample positions for given set of pieces on board.
        To generate more realisticgame positions, one could account 
        for different factors, such as correlation 
        between a type of the piece and its expected 
        position, removing ilegal positions etc.
        """
        n_pieces_on_board = len(pieces)
        pos_coordinates = np.random.choice(
                                    np.arange(const.N_SQUARES), 
                                    size=n_pieces_on_board, 
                                    replace=False)
        return self._coordinates[pos_coordinates]

    @staticmethod
    def add_alpha_channel(img):
        if img.shape[2] < 4:
          alpha_ch = np.ones_like(img[...,0], dtype=np.uint8)*255
          img = np.concatenate([img, alpha_ch[...,None]], axis=-1)
        return img

    def _generate_board_and_encoding( 
            self, 
            pieces_set: str, 
            board_im: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates image of a random chess position and the corresponding encoding
        """
        pieces = self._sample_pieces()
        pos_coordinates = self._sample_positions(pieces)
        position_image = cv2.imread(board_im, cv2.IMREAD_COLOR)
        position_image = self.add_alpha_channel(position_image)
        position_encoding = np.zeros(
                      [const.BOARD_RANKS, const.BOARD_FILES], dtype=np.int32)

        for piece_pos, piece in zip(pos_coordinates, pieces):
            rank, file = piece_pos 
            piece_image = os.path.join(pieces_set, f'{piece}.png')
            piece_image = cv2.imread(piece_image, cv2.IMREAD_UNCHANGED)
            position_image = fen_transcode.place_piece_on_board(
                                position_image, 
                                piece_image, 
                                file, 
                                rank)
            position_encoding[rank, file] = const.PIECES_ENCODING[piece]

        out_dim = random.randint(self.output_min_size, self.output_max_size)
        position_image = cv2.resize(position_image, (out_dim, out_dim))
        return position_image, position_encoding

    def _get_background(self):
        background_img = random.choice(self.backgrounds)
        image = cv2.imread(background_img, cv2.IMREAD_COLOR)
        image = self.add_alpha_channel(image)
        image = cv2.resize(image, (self.BACKGROUND_W, self.BACKGROUND_H))
        return image


    def _get_crop_offsets(self, backgr, board, board_y_offset, board_x_offset):
        """
        Takes the full image and crop out the board. Based on the _only_boards, the 
        maximum possible offsets are decided.
        """

        if self._only_boards:
            # Some boards are generated with large offset, some with smaller/none.
            hard_or_ezy_switch = np.random.randint(0, 2)
            ezy_off = 0
            hard_off = 1/16

            if hard_or_ezy_switch == 0:
                y_off = random.randint(0, round(ezy_off*board.shape[0]))
                x_off = random.randint(0, round(ezy_off*board.shape[1]))
                h_add = y_off + random.randint(0, round(ezy_off*board.shape[0]))
                w_add = x_off + random.randint(0, round(ezy_off*board.shape[1]))
            else:
                y_off = random.randint(0, round(hard_off*board.shape[0]))
                x_off = random.randint(0, round(hard_off*board.shape[1]))
                h_add = y_off + random.randint(0, round(hard_off*board.shape[0]))
                w_add = x_off + random.randint(0, round(hard_off*board.shape[1]))
        else:
            y_off = random.randint(0, board_y_offset)
            x_off = random.randint(0, board_x_offset)
            h_add = y_off + random.randint(0, (backgr.shape[0] - board.shape[0] - board_y_offset))
            w_add = x_off + random.randint(0, (backgr.shape[1] - board.shape[1] - board_x_offset))

        return y_off, x_off, h_add, w_add

    def _create_board(self, backgr, board):
        #Puts board in the middle
        board_y_offset = random.randint(0, (backgr.shape[0]-board.shape[0]))
        board_x_offset = random.randint(0, (backgr.shape[1]-board.shape[1]))

        image = fen_transcode.overlay_png_images(
            backgr, board, 
            board_y_offset, 
            board_x_offset)

        y_off, x_off, h_add, w_add = self._get_crop_offsets(
            backgr, 
            board, 
            board_y_offset, 
            board_x_offset)

        y_crop = max(0, board_y_offset - y_off)
        x_crop = max(0, board_x_offset - x_off)
        h_crop = board.shape[0] + h_add
        w_crop = board.shape[1] + w_add

        # board relative cooridantes
        t_y = y_off/h_crop - 1/2
        t_x = x_off/w_crop - 1/2
        t_h = np.log(board.shape[0]/h_crop)
        t_w = np.log(board.shape[1]/w_crop)

        crop_img = image[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        crop_img = cv2.resize(crop_img, (self.out_board_W, self.out_board_H))
        return crop_img, (t_y, t_x, t_h, t_w)

    def _generator(self):
        """
        Python generator returning random chessboards with background
        """
        while True:
            # Take random piece style and board style
            pieces_set = np.random.choice(self.pieces_sets)
            board_im = np.random.choice(self.boards)
            # Create the chess board itself, together with the gold label
            image, label = self._generate_board_and_encoding(pieces_set, board_im)
            # Flip the label to "human-natural" orientation
            label = np.flip(label, axis = 0)
            # Add background
            board_with_backgroud, bounding_box = self._create_board(
                                                  self._get_background(), image)
            # Remove the alpha channel
            board_with_backgroud = board_with_backgroud[...,:-1]
            yield (board_with_backgroud, label, bounding_box)

    def tfGenerator(self, only_boards: bool = True) -> tf.data.Dataset:
        """
        Creates the tensorflow dataset generator, using the python 
        generator self._generator()
        """
        self._only_boards = only_boards
        data_generator = tf.data.Dataset.from_generator(
        self._generator, 
          output_types=(tf.uint8, tf.int32, tf.float32), 
          output_shapes=([self.out_board_W, self.out_board_H, self.COL_CHANNELS], 
                          [const.BOARD_RANKS, const.BOARD_FILES], 
                          [4])
        )
        return data_generator