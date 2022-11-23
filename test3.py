import pandas as pd
import ProjectAssessment as pa

def main():
    print("Dataset Two with Added Columns")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'])

    print("Dataset Two with Added Columns with Rubric Bootstrapping")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], rubric=True)

    print("Dataset Two with Added Columns in Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], linear=True)

    print("Dataset Two with Added Columns")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['d',])

    print("Dataset Two with Added Columns in Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t',], linear=True)

if __name__ == '__main__':
    main()