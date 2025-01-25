import pandas as pd
import numpy as np

from ga import eval_on_bracket

def csv_to_nd_array(path: str):
    """
    Reads in a csv with 63 columns and n rows, 
    and returns a (n, 63) numpy array

    Parameters
    ----------
    path: str
        path to file location
    
    Returns
    -------
        A (n, 63) numpy array with the same data
    """
    submission = pd.read_csv(path, header=None).to_numpy()
    # assert submission.shape[1] == 63
    print(submission.shape)
    return submission

def make_test(n_rows: int):
    test_df = np.random.randint(0, 2, (n_rows, 63))
    test_df = pd.DataFrame(test_df)
    print(test_df)
    test_df.to_csv("../data/submissions/test.csv", index=False, header=False)

    test_true = np.random.randint(0, 2, (1, 63))
    test_true = pd.DataFrame(test_true)
    print(test_true)
    test_true.to_csv("../data/submissions/test_true.csv", index=False, header=False)

def main():
    sub1 = csv_to_nd_array("../data/submissions/test.csv")
    true = csv_to_nd_array("../data/submissions/test_true.csv")

    print(eval_on_bracket(true, sub1.reshape((1, -1, 63))))
    for i in range(sub1.shape[0]):
        print(i, sub1.shape[0])
        print(eval_on_bracket(true, sub1.reshape((1, -1, 63))[:, i:i+1, :]))


if __name__ == "__main__":
    np.random.seed(42)
    make_test(5)
    main()
