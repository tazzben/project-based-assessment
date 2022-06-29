import pandas as pd
import ProjectAssessment as pa

def main():
    print("Dataset One")
    df = pd.read_csv('data.csv')
    pa.DisplayResults(df)

    print("Dataset One with Linear Model")
    df = pd.read_csv('data.csv')
    pa.DisplayResults(df, linear=True)

    print("Dataset Two")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df)

    print("Dataset Two with Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, linear=True)

    print("Dataset One with Bootstrapped Rubric")
    df = pd.read_csv('data.csv')
    pa.DisplayResults(df, rubric=True)

    print("Dataset Two with Bootstrapped Rubric")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, rubric=True)

if __name__ == '__main__':
    main()
