import pandas as pd
import ProjectAssessment as pa

def main():
    # print("Dataset Large")
    # df = pd.read_csv('large.csv')
    # pa.DisplayResults(df)
    # pa.SaveResults(df)
    print("Dataset Small with Columns")
    df = pd.read_csv('data3.csv')
    pa.DisplayResults(df, columns=['const','s12'], no_students=True )
    pa.DisplayResults(df, columns=['const','s12'], no_questions=True )
    pa.DisplayResults(df, columns=['const','s12'], no_questions=True, no_students=True )

if __name__ == '__main__':
    main()