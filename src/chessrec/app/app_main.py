#!/usr/bin/env python3
import os
import sys

import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np 
import argparse

from chessrec.engine_interface.stockfish.interface import EngineInterface
from chessrec.models.board_detector_v0 import BoardDetector
from chessrec.models.position_recognizer_v0 import PositionRecognizer
import chessrec.fen_transcode as fen_transcode
import chessrec.app.app_buttons as butt

# TODO: add annotations everywhere

class ChessEvalApp():
    def __init__(self, master: tk.Tk, detector: BoardDetector, recognizer: PositionRecognizer, args: argparse.Namespace):
        # Interface used to suggest best lines for a given FEN
        self.engine_interface = EngineInterface(
            args.stockfish_elo,
            args.stockfish_hash,
            args.stockfish_threads,
            args.stockfish_depth,
        )
        # ML model used to recognize pieces on the board and their position
        self.recognizer = recognizer
        # Area on the display to screenshot chess board from
        self.screenshot_area = None

        # tkinter master
        self.master = master
        self.master.title("Chess evaluation App")
        self.master_W = self.master.winfo_width()
        self.master.bind("<Configure>", self.on_resize)
        self.message_text = tk.Text(self.master, width=self.text_window_width)
        self._text_insert_mode = '1.0'

        # Evaluation button - takes screenshot and feeds it to the recognizer and the chess engine
        self.eval_button = tk.Button(
            self.master, 
            text="Evaluate", 
            command=self.eval_position)

        # Other independent buttons: Screenshot are selection, casting options, board orientation...
        self.select_button = butt.SelectButton(self, detector, button_to_disable=self.eval_button)
        self.player_perspective_buttons = butt.PlayerPerspectiveButtons(self)
        self.player_on_move_buttons = butt.PlayerOnMoveButtons(self)
        self.castling_buttons = butt.CastlingButtons(self)

        # Images of the screenshot and the reconstructed board
        self.captured_pos_IM = Image.new('RGB', self.img_sizes)
        self.captured_pos_tk = ImageTk.PhotoImage(self.captured_pos_IM)
        self.captured_pos_label = tk.Label(self.master)
        self.decoded_pos_IM = Image.new('RGB', self.img_sizes)
        self.decoded_pos_tk = ImageTk.PhotoImage(self.decoded_pos_IM)
        self.decoded_pos_label = tk.Label(self.master)
        
        # Placing of the buttons and images in the app window
        self._initialize_grid()

    @property
    def img_sizes(self):
        # Definition of the relative sizes of the images
        H = self.master.winfo_height()//3
        W = self.master.winfo_width()//2
        return (W, H)

    @property
    def text_window_width(self):
        # Definition of the relative size of the text window
        W = self.master.winfo_width()//9
        return W
    
    def _initialize_grid(self) -> None:
        self.eval_button.grid(row=0, column=0)
        self.select_button.select_button.grid(row=0, column=1)
        self.player_perspective_buttons.play_as_white_check.grid(row=1, column=0)
        self.player_perspective_buttons.play_as_black_check.grid(row=1, column=1)
        self.player_on_move_buttons.white_on_move_button.grid(row=2, column=0)
        self.player_on_move_buttons.black_on_move_button.grid(row=2, column=1)
        self.castling_buttons.B_OO_button.grid(row=3, column=0)
        self.castling_buttons.B_OOO_button.grid(row=3, column=1)
        self.castling_buttons.W_OO_button.grid(row=4, column=0)
        self.castling_buttons.W_OOO_button.grid(row=4, column=1)
        self.captured_pos_label.grid(row=5, column=0, sticky="nsew")
        self.decoded_pos_label.grid(row=5, column=1, sticky="nsew")
        self.message_text.grid(row=6, column=0, columnspan=2)

    def on_resize(self, event):
        # Resize the image. Note that the original image is not overwritten
        # to maintain the original quality of the image
        self.decoded_pos_tk  = ImageTk.PhotoImage(
            self.decoded_pos_IM.resize(self.img_sizes)
        )
        self.captured_pos_tk = ImageTk.PhotoImage(
            self.captured_pos_IM.resize(self.img_sizes)
        )
        self.decoded_pos_label.config(image=self.decoded_pos_tk)
        self.captured_pos_label.config(image=self.captured_pos_tk)
        # Resize message window
        self.message_text.configure(width=self.text_window_width)

    def update_images(self, screenshot, decoded_pos):
        # Resize the screenshot and decoded position images and update the GUI labels
        self.captured_pos_IM = screenshot.resize(self.img_sizes)
        self.decoded_pos_IM = decoded_pos.resize(self.img_sizes)
        self.captured_pos_tk = ImageTk.PhotoImage(self.captured_pos_IM)
        self.decoded_pos_tk = ImageTk.PhotoImage(self.decoded_pos_IM)
        self.captured_pos_label.config(image=self.captured_pos_tk)
        self.decoded_pos_label.config(image=self.decoded_pos_tk)

    def recognize_and_decode_position(self, screenshot_area, play_as_white: bool):
        # Takes screenshot and passes it to the recognizer
        with ImageGrab.grab(bbox=screenshot_area) as screenshot:
            screenshot = screenshot.resize((256, 256))
            image = np.array(screenshot.convert('L'), dtype=np.float32)
        encoded_pos = self.recognizer.predict(image[None,...,None])[0]
        encoded_pos = encoded_pos._numpy().astype(np.uint8)
        # Check the board orientation
        if not play_as_white:
            encoded_pos = np.flip(encoded_pos, axis=0)
            encoded_pos = np.flip(encoded_pos, axis=1)
        # Reconstruct the image
        decoded_pos = fen_transcode.decode_position(encoded_pos)
        decoded_pos = Image.fromarray(decoded_pos)
        return screenshot, decoded_pos, encoded_pos

    def get_FEN(self, encoded_pos):
        # Translate the cathegorical encoding to FEN notation
        postition_FEN = fen_transcode.encoding_to_FEN(encoded_pos)
        player_on_move_FEN = self.player_on_move_buttons.player_on_move
        castling_FEN = self.castling_buttons.castling_FEN_part
        return f'{postition_FEN} {player_on_move_FEN} {castling_FEN} - 0 1'

    def eval_position(self):
        # Prevent to evaluate before selecting the board 
        if self.screenshot_area == None:
            self.message_text.insert(
                self._text_insert_mode, f'Select area to capture first. \n')
            return
        else:
            screenshot, decoded_pos, encoded_pos = self.recognize_and_decode_position(
                        self.screenshot_area,
                        self.player_perspective_buttons.play_as_white
            )
            self.update_images(screenshot, decoded_pos)
            FEN = self.get_FEN(encoded_pos)
            self.message_text.insert(
                self._text_insert_mode, f'______________\n')
            self.message_text.insert(
                self._text_insert_mode, f'FEN: {FEN}\n')
            engine_report = self.engine_interface.stockfish_evaluation(FEN)
            self.message_text.insert(self._text_insert_mode, engine_report)
