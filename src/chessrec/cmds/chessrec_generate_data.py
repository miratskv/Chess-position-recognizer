#!/usr/bin/env python3

import os 
import argparse 
import pkg_resources

from chessrec.data_generator import ChessBoardGenerator
app_path = pkg_resources.resource_filename('chessrec', "")
example_assets = os.path.join(app_path, "cmds", "generator_assets_example")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_size", default=10, type=int, help="Size of dataset")
parser.add_argument("--save_path", default="generated_data", type=str, help="Path to save the dataset")
parser.add_argument("--only_boards", default=True, type=bool, 
    help="Two modes: Generate with full background or with only small background edges")
parser.add_argument("--piece_sets_path", default=os.path.join(example_assets, "chess_pieces"), type=str, help="Path to piece sets")
parser.add_argument("--boards_imgs_path", default=os.path.join(example_assets, "chess_boards"), type=str, help="Path to boards")
parser.add_argument("--background_im_path", default=os.path.join(example_assets, "backgrounds"), type=str, help="Path to background images")


def main() -> None:
    args = parser.parse_args([] if "__file__" not in globals() else None)
    generator =  ChessBoardGenerator(
            args.boards_imgs_path, 
            args.piece_sets_path,
            args.background_im_path).tfGenerator(only_boards=args.only_boards)
    data = generator.take(args.dataset_size)
    data.save(args.save_path)

if __name__ == '__main__':
    main()
