import pandas as pd
import ProjectAssessment as pa

def main():
    print("Dataset One with Linear Model")
    df = pd.read_csv('data.csv')
    pa.SaveResults(df, linear=True)
    print("Dataset One with Non-Linear Model")
    pa.SaveResults(df, rubricFile='RubricNonLinear.csv', studentFile='StudentNonLinear.csv', outputFile='OutputNonLinear.csv')

if __name__ == '__main__':
    main()