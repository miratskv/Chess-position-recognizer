#!/usr/bin/env python3

from chessrec.app.app_main import ChessEvalApp
from chessrec.models.board_detector_v0 import BoardDetector
from chessrec.models.position_recognizer_v0 import PositionRecognizer

import os
import pkg_resources
import argparse
import tkinter as tk

parser = argparse.ArgumentParser()
app_path = pkg_resources.resource_filename('chessrec', "")
default_weights_dir = os.path.join(app_path, "models", "trained_weights")

parser.add_argument("--master_H", default=540, type=int, help="App window default height")
parser.add_argument("--master_W", default=360, type=int, help="App window default width")

parser.add_argument("--stockfish_elo", default=3000, type=int, help="")
parser.add_argument("--stockfish_hash", default=2048, type=int, help="")
parser.add_argument("--stockfish_threads", default=4, type=int, help="")
parser.add_argument("--stockfish_depth", default=16, type=int, help="") 

# TODO: Changing model would actually require to also change the source of the PositionRecognizer etc. !!
# add some API to do this comfortably...
parser.add_argument(
    "--load_detector", 
    default=os.path.join(default_weights_dir, "board_detector_v0.h5"), 
    type=str, 
    help="Path to detector weights"
)
parser.add_argument(
    "--load_recognizer", 
    default=os.path.join(default_weights_dir, "position_recognizer_v0.h5"), 
    type=str, 
    help="Path to recognizer weights")

def main() -> None:
    args = parser.parse_args([] if "__file__" not in globals() else None)
    detector = BoardDetector()
    recognizer = PositionRecognizer()
    
    detector.load_model(os.path.join(app_path, args.load_detector))
    recognizer.load_model(os.path.join(app_path, args.load_recognizer))

    root = tk.Tk()
    root.geometry(f'{args.master_W}x{args.master_H}')
    ChessEvalApp(root, detector, recognizer, args)
    root.mainloop()

if __name__ == '__main__':
    main()

