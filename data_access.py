import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pickle
from config import seed,raw_data_file,data_store 
import re



def print_na_summary(name, df):
    na_summary = pd.DataFrame({
        "NA_count": df.isna().sum(),
        "NA_percent": df.isna().mean() * 100
    })
    na_summary = na_summary[na_summary["NA_count"]>0]
    # 按缺失数量降序排列
    na_summary = na_summary.sort_values(
        "NA_count", ascending=False
    )

    print(f"\n{name}:")
    print(na_summary.to_string(
        float_format=lambda x: f"{x:.2f}%"
    ))




def main():
    raw_connectome = pd.read_csv(raw_data_file/"TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
    raw_solu = pd.read_excel(raw_data_file/"TRAINING_SOLUTIONS.xlsx")
    raw_categorical = pd.read_excel(raw_data_file/"TRAIN_CATEGORICAL_METADATA_new.xlsx")
    raw_quan = pd.read_excel(raw_data_file/"TRAIN_QUANTITATIVE_METADATA_new.xlsx")

    raw_id_is_equal = (np.array_equal(raw_categorical["participant_id"].to_numpy(),raw_connectome["participant_id"].to_numpy())
                    and np.array_equal(raw_categorical["participant_id"].to_numpy(),raw_solu["participant_id"].to_numpy())
                    and np.array_equal(raw_categorical["participant_id"].to_numpy(),raw_quan["participant_id"].to_numpy())
                )
    print(f"Raw participants ID across raw data is identical: {raw_id_is_equal}")

    id_ref_order = raw_connectome["participant_id"]
    raw_quan_aligned = (raw_quan.set_index("participant_id")
                        .loc[id_ref_order]
                        .reset_index())
    raw_categorical_aligned = (raw_categorical.set_index("participant_id")
                        .loc[id_ref_order]
                        .reset_index())
    raw_solu_aligned = (raw_solu.set_index("participant_id")
                        .loc[id_ref_order]
                        .reset_index())

    Aligned_id_is_equal = (np.array_equal(raw_categorical_aligned["participant_id"].to_numpy(),raw_connectome["participant_id"].to_numpy())
                    and np.array_equal(raw_categorical_aligned["participant_id"].to_numpy(),raw_solu_aligned["participant_id"].to_numpy())
                    and np.array_equal(raw_categorical_aligned["participant_id"].to_numpy(),raw_quan_aligned["participant_id"].to_numpy())
                )
    print(f"Alighed participants ID across raw data is identical: {Aligned_id_is_equal}")
    assert Aligned_id_is_equal, "Participant id not equal!"

    test_sample = raw_solu.iloc[[14]]
    test_id = test_sample['participant_id'].iloc[0]
    target_sample = raw_solu_aligned[raw_solu_aligned["participant_id"] == test_id]

    print(f"test_sample:{test_sample}")
    print(f"target_sample:{target_sample}")
    print("Test ID: Connectome")
    print(np.where(raw_connectome["participant_id"] == test_id)[0])
    if np.array_equal(test_sample.to_numpy(),target_sample.to_numpy()):
        print ("Solu Pass")
    print(np.where(raw_solu_aligned["participant_id"] == test_id)[0])

    test_cate = raw_categorical[raw_categorical["participant_id"] == test_id]
    target_cate = raw_categorical_aligned[raw_categorical_aligned["participant_id"] == test_id]
    if np.array_equal(test_cate.to_numpy(),target_cate.to_numpy()):
        print ("Categorical Pass")
    print(np.where(raw_categorical_aligned["participant_id"] == test_id))

    test_quan = raw_quan[raw_quan["participant_id"] == test_id]
    target_quan = raw_quan_aligned[raw_quan_aligned["participant_id"] == test_id]
    if np.array_equal(test_quan.to_numpy(),target_quan.to_numpy()):
        print ("Quan Pass")
    print(np.where(raw_quan_aligned["participant_id"] == test_id))

    print_na_summary("raw_categorical_aligned", raw_categorical_aligned)
    print_na_summary("raw_quan_aligned", raw_quan_aligned)
    print_na_summary("raw_solu_aligned", raw_solu_aligned)
    print_na_summary("raw_connectome", raw_connectome)

    y_adhd = raw_solu_aligned["ADHD_Outcome"]
    y_gender = raw_solu_aligned["Sex_F"]

    demography_data = pd.concat([raw_quan_aligned,raw_categorical_aligned.iloc[:,2:]],axis=1)
    age_by_site = demography_data.groupby("Basic_Demos_Study_Site")["MRI_Track_Age_at_Scan"].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        missing=lambda x: x.isna().sum(),
        missing_pct=lambda x: x.isna().mean()*100
    )
    print(age_by_site.to_string(
        float_format=lambda x: f"{x:.2f}"
    ))



    train_idx, test_idx = train_test_split(
        np.arange(len(demography_data)), 
        test_size=0.2, 
        random_state=seed, 
        stratify=y_gender
    )

    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=0.2, 
        random_state=seed, 
        stratify=y_gender.iloc[train_idx]
    )

    train_id = id_ref_order.iloc[train_idx].tolist()
    val_id = id_ref_order.iloc[val_idx].tolist()
    test_id = id_ref_order.iloc[test_idx].tolist()


    print(demography_data["PreInt_Demos_Fam_Child_Ethnicity"].unique())
    demography_data_imputed = demography_data.copy()
    demography_data_imputed["PreInt_Demos_Fam_Child_Ethnicity"] = demography_data_imputed["PreInt_Demos_Fam_Child_Ethnicity"].fillna(3)
    imputer_age = SimpleImputer(strategy="median").set_output(transform="pandas")

    age_col = ["MRI_Track_Age_at_Scan"]
    age_nan = demography_data[age_col].isna().astype(int)
    demography_data_imputed["age_missing"] = age_nan
    print(age_nan)
    imputer_age.fit(demography_data_imputed.iloc[train_idx][age_col])
    demography_data_imputed[age_col] = imputer_age.transform(demography_data_imputed[age_col])

    # TRAIN_NEW keeps missing questionnaire values as NaN (the old data had them as 0).
    # Fill the other quantitative columns with their training-set median.
    # Categorical NaNs are left as they are: OneHotEncoder(handle_unknown="ignore") encodes them as all zeros.
    numeric_cols = [c for c in raw_quan_aligned.columns if c not in ["participant_id", *age_col]]
    imputer_numeric = SimpleImputer(strategy="median").set_output(transform="pandas")
    imputer_numeric.fit(demography_data_imputed.iloc[train_idx][numeric_cols])
    demography_data_imputed[numeric_cols] = imputer_numeric.transform(demography_data_imputed[numeric_cols])
    assert not demography_data_imputed[numeric_cols + age_col].isna().any().any(), "NaN left in numeric columns"
    print_na_summary("demography_data_imputed", demography_data_imputed)


    with open(data_store/"demography_data_imputed.pkl", 'wb') as f:
        pickle.dump ({
            'demography_data_imputed':demography_data_imputed,
            'imputer_age':imputer_age,
            'imputer_numeric':imputer_numeric
        },f)

    with open(data_store/"fMRI_Connectome_data.pkl", 'wb') as f:
        pickle.dump ({
            'fMRI_Connectome':raw_connectome
        },f)

    with open(data_store/"split_indices.pkl", 'wb') as f:
        pickle.dump ({
            'train_idx':train_idx,
            'test_idx':test_idx,
            'val_idx':val_idx,
            'train_id':train_id,
            'val_id':val_id,
            'test_id':test_id
        },f)

    with open(data_store/"y_results.pkl", 'wb') as f:
        pickle.dump ({
            'y_adhd':y_adhd,
            'y_gender':y_gender
        },f)


    dictionary = pd.read_excel(
        raw_data_file.parent/"Data Dictionary.xlsx",
        sheet_name="Dictionary",
    )

    cate_rows = dictionary.loc[
        dictionary['DataType'].str.strip().eq('Categorical'),
        ['Field','Labels']
    ]

    print(cate_rows['Field'])
    cate_rows = cate_rows.drop(index=23)

    categorical_values = {}
    for row in cate_rows.itertuples():
        field_name = row.Field.strip()
        labels = row.Labels.replace("\\n",'\n')
        codes = re.findall(
            r"(?m)^\s*(\d+)\s*=",
            labels
        )
        codes = [
            int(code)
            for code in codes
        ]
        categorical_values[field_name] = codes

    with open(data_store/"categorical_values.pkl", 'wb') as f:
        pickle.dump (categorical_values,f)

if __name__ == "__main__":
    main()