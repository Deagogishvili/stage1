import pandas as pd

def process_tmhmm(TMHMM_PATH, PROCESSED_DATA_PATH, threshold):

    df = pd.read_csv(TMHMM_PATH,  sep='\t',  header=None)
    expAA = df[2].str.split('=',expand=True)[1]
    greater_than_18 =  pd.to_numeric(expAA) >= threshold

    df_tmhmm = pd.DataFrame({'id':df[0]})
    df_tmhmm.loc[greater_than_18, 'tmp'] = True
    df_tmhmm.loc[~greater_than_18, 'tmp'] = False
    df_tmhmm.to_csv(PROCESSED_DATA_PATH,  index=False)

if __name__ == '__main__':
    import yaml
    config = yaml.safe_load(open("../config.yml"))

    process_tmhmm(config['path']['TMHMM']+'webface2.fcgi?jobid=5E1658FE00002E03F382FA54', config['path']['processed_data']+'tmhmm_data.csv', 18)
