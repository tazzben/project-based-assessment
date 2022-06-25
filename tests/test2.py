import pandas as pd
import ProjectAssessment as pa

def main():
    print("Dataset One with Linear Model")
    df = pd.read_csv('data.csv')
    pa.DisplayResults(df, linear=True)

if __name__ == '__main__':
    main()