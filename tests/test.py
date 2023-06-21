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

    print("Dataset Two with One Added Columns")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t',])

    print("Dataset Two with One Added Columns")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['d',])

    print("Dataset Two with Added Columns")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'])

    print("Dataset Two with Added Columns with Rubric Bootstrapping")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], rubric=True)

    print("Dataset Two with Added Columns with no_s on")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_students=True)

    print("Dataset Two with Added Columns with no_q on")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_questions=True)

    print("Dataset Two with Added Columns with no_s and no_q on")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_students=True, no_questions=True)

    print("Dataset Two with Added Columns with no_s and no_q on and rubric bootstrapping")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_students=True, no_questions=True, rubric=True)

    print("Dataset Two with Added Columns with no_s on - Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_students=True, linear=True)

    print("Dataset Two with Added Columns with no_q on - Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_questions=True, linear=True)

    print("Dataset Two with Added Columns with no_s and no_q on - Linear Model")
    df = pd.read_csv('data2.csv')
    pa.DisplayResults(df, columns=['t','d'], no_students=True, no_questions=True, linear=True)

    print("Dataset Two with Added Columns with no_s and no_q on and rubric bootstrapping - Linear Model")
    df = pd.read_csv('data2.csv')   
    pa.DisplayResults(df, columns=['t','d'], no_students=True, no_questions=True, rubric=True, linear=True)

if __name__ == '__main__':
    main()
