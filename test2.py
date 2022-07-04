import pandas as pd
import ProjectAssessment as pa

def main():
    print("Dataset One with Linear Model")
    df = pd.read_csv('data2.csv')
    pa.SaveResults(df, linear=True)

if __name__ == '__main__':
    main()