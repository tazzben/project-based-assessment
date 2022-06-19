import ProjectAssessment as pa
import pandas as pd

def main():
    df = pd.read_csv('data2.csv')
    pa.SaveResults(df, rubric=False)

if __name__ == '__main__':
    main()