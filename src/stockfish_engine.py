import chess
import chess.engine
import pandas as pd

def get_best_move(fen, engine, time_limit=2.0):
    board = chess.Board(fen)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move.uci(), board.san(result.move)

def validate_puzzle(predicted_fen, expected_move_uci, engine, time_limit=2.0):
    board = chess.Board(predicted_fen)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move.uci() == expected_move_uci

def get_puzzle_position_and_answer(fen, moves_str):
    moves = moves_str.split()
    board = chess.Board(fen)
    # Apply opponent's move to get the puzzle position
    board.push_uci(moves[0])
    # The second move is the correct answer
    correct_move = moves[1]
    return board.fen(), correct_move

if __name__ == "__main__":
    engine = chess.engine.SimpleEngine.popen_uci(r"../engine/stockfish/stockfish-windows-x86-64-avx2.exe")

    # Load puzzles
    puzzles = pd.read_csv(
        r"C:\Users\kemal\IdeaProjects\ChessVisionProject\data\lichess_db_puzzle.csv",
        names=["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
               "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]
    )

    # Sample a small subset to test
    sample = puzzles.sample(n=100, random_state=42)

    correct = 0
    for _, row in sample.iterrows():
        puzzle_fen, expected_move = get_puzzle_position_and_answer(row["FEN"], row["Moves"])
        uci, san = get_best_move(puzzle_fen, engine)

        match = uci == expected_move
        if match:
            correct += 1

        print(f"Puzzle {row['PuzzleId']}: Expected {expected_move}, Got {uci} — {'✓' if match else '✗'}")

    print(f"\nAccuracy: {correct}/{len(sample)} ({100 * correct / len(sample):.1f}%)")

    engine.quit()