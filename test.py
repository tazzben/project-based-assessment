import ProjectAssessment as pa
import pandas as pd

def main():
    df = pd.read_csv('data.csv')
    pa.DisplayResults(df, rubric=True)

if __name__ == '__main__':
    main()