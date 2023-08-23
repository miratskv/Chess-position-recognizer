import tkinter as tk
from collections import OrderedDict
import numpy as np
from PIL import ImageGrab


class SelectButton():
    def __init__(self, parent, detector, button_to_disable=None):
        self.parent = parent
        self.detector = detector

        self.select_button = tk.Button(
                                self.parent.master, 
                                text="Select Area", 
                                command=self.select_capture_area)
        self.button_to_disable = button_to_disable 
        self.selection_in_progress = False
        self.root = None

    def start_root(self, window_width, window_height, window_x, window_y):

        self.select_button.configure(
                text="End Selection", 
                command=self.end_selection
        )
        if self.button_to_disable:
            self.button_to_disable.configure(state='disable')
        self.selection_in_progress = True

        self.root = tk.Tk()
        self.root.wait_visibility(self.root)
        self.root.wm_attributes('-alpha', 0.5)
        self.root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
        self.root.title("Capture area")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.selection_in_progress = False
        self.select_button.configure(
                text="Select Area", 
                command=self.select_capture_area
        )
        self.root.destroy()
        if self.button_to_disable:
            self.button_to_disable.configure(state='normal')

    def end_selection(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        self.parent.screenshot_area = (x, y, x+width, y+height)
        self.on_closing()

    def select_capture_area(self):
        if not self.selection_in_progress:

            #self.eval_button.configure(state='disabled')
            with ImageGrab.grab() as screenshot:
                screenshot = screenshot.resize((256, 256)).convert('L')
                image = np.array(screenshot, dtype=np.float32)
                image = np.expand_dims(image, axis=0)
                image = np.expand_dims(image, axis=3)

            board, board_detec_coor = self.detector.predict(image)
            board_detec_coor = board_detec_coor[0]._numpy()
            screen_width = self.parent.master.winfo_screenwidth()
            screen_height = self.parent.master.winfo_screenheight()
            window_y = round(screen_height * board_detec_coor[0])
            window_x = round(screen_width * board_detec_coor[1])
            window_height =  round(screen_height * board_detec_coor[2]) - window_y
            window_width = round(screen_width * board_detec_coor[3]) - window_x

            self.start_root(window_width, window_height, window_x, window_y)


class CastlingButtons():
    def __init__(self, parent):
        self.castling_is_possible = OrderedDict(K=True, Q=True, k=True, q=True)
        switch_key_value = lambda dict_, key: dict_.__setitem__(key, not dict_.get(key, False))
        
        self.B_OO_variable = tk.BooleanVar(value=True)
        self.B_OOO_variable = tk.BooleanVar(value=True)
        self.W_OO_variable = tk.BooleanVar(value=True)
        self.W_OOO_variable = tk.BooleanVar(value=True)

        self.B_OO_button = tk.Checkbutton(
                                parent.master, 
                                text="B-OO", 
                                variable=self.B_OO_variable, 
                                command=lambda: switch_key_value(self.castling_is_possible, "k"))
        self.B_OOO_button = tk.Checkbutton(
                                parent.master, 
                                text="B-OOO", 
                                variable=self.B_OOO_variable, 
                                command=lambda: switch_key_value(self.castling_is_possible, "q"))
        self.W_OO_button = tk.Checkbutton(
                                parent.master, 
                                text="W-OO", 
                                variable=self.W_OO_variable, 
                                command=lambda: switch_key_value(self.castling_is_possible, "K"))
        self.W_OOO_button = tk.Checkbutton(
                                parent.master, 
                                text="W-OOO", 
                                variable=self.W_OOO_variable, 
                                command=lambda: switch_key_value(self.castling_is_possible, "Q"))
    @property
    def castling_FEN_part(self):
        castling_FEN = ''.join(
                [key for key, value in self.castling_is_possible.items() if value])
        return castling_FEN


class PlayerOnMoveButtons():
    def __init__(self, parent):
        self.parent = parent
        self.player_on_move = "w"
        self.white_on_move_var = tk.BooleanVar(value=True)
        self.black_on_move_var = tk.BooleanVar(value=False)
        self.white_on_move_button = tk.Checkbutton(
                                        parent.master, 
                                        text="White to move", 
                                        variable=self.white_on_move_var, 
                                        command=self.change_player)

        self.black_on_move_button = tk.Checkbutton(
                                        parent.master, 
                                        text="Black to move", 
                                        variable=self.black_on_move_var, 
                                        command=self.change_player)

    def change_player(self):
        self.player_on_move = "b" if self.player_on_move == "w" else "w"
        if self.player_on_move == "w":
            self.white_on_move_var.set(True)
            self.black_on_move_var.set(False)
        if self.player_on_move == "b":
            self.white_on_move_var.set(False)
            self.black_on_move_var.set(True)
        self.parent.message_text.insert(
            self.parent._text_insert_mode, f'Player to move: {self.player_on_move} \n')


class PlayerPerspectiveButtons():
    def __init__(self, parent):
        self.parent = parent
        self.play_as_white = True
        self.play_as_white_var = tk.BooleanVar(value=True)
        self.play_as_black_var = tk.BooleanVar(value=False)
        self.play_as_white_check = tk.Checkbutton(
                                        parent.master, 
                                        text="Play as white", 
                                        variable=self.play_as_white_var, 
                                        command=self.change_board_orientation)

        self.play_as_black_check = tk.Checkbutton(
                                        parent.master, 
                                        text="Play as black", 
                                        variable=self.play_as_black_var, 
                                        command=self.change_board_orientation)

    def change_board_orientation(self):
        self.play_as_white = not self.play_as_white
        if self.play_as_white:
            self.play_as_white_var.set(True)
            self.play_as_black_var.set(False)
            self.parent.message_text.insert(
                self.parent._text_insert_mode, "Player perspective: w \n")
        else:
            self.play_as_white_var.set(False)
            self.play_as_black_var.set(True)
            self.parent.message_text.insert(
                self.parent._text_insert_mode, "Player perspective: b \n") 
