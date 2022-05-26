import pandas as pd


def divide_train_test(info_type):
    """
    :param info_type: star or credit
    :return: None
    """
    if info_type == "star":
        df = pd.read_csv("../data/csv_data/pri_star_info.csv").fillna("-1")
        train = df[(df["star_level"] != -1)]
        test = df[(df["star_level"] == -1)]
        print(train)
        print(test)
        train.to_csv("../data/processed_data/star_train.csv", index=False, mode='w', line_terminator='\n',
                     encoding='utf-8')
        test.to_csv("../data/processed_data/star_test.csv", index=False, mode='w', line_terminator='\n',
                    encoding='utf-8')
    elif info_type == "credit":
        df = pd.read_csv("../data/csv_data/pri_credit_info.csv").fillna("-1")
        train = df[(df["credit_level"] != -1)]
        test = df[(df["credit_level"] == -1)]
        print(train)
        print(test)
        train.to_csv("../data/processed_data/credit_train.csv", index=False, mode='w', line_terminator='\n',
                     encoding='utf-8')
        test.to_csv("../data/processed_data/credit_test.csv", index=False, mode='w', line_terminator='\n',
                    encoding='utf-8')


def append_star_other_info(star_path):
    df = pd.read_csv(star_path)
    df.set_index("uid", inplace=True)
    print(df)

    cust_base = pd.read_csv("../data/csv_data/pri_cust_base_info.csv")[
        ["uid", "sex", "marrige", "is_black", "is_contact"]
    ].drop_duplicates()
    cust_base.set_index("uid", inplace=True)
    print(cust_base)
    df = df.join(cust_base, on="uid", how="left")
    print(df)

    cust_asset = pd.read_csv("../data/csv_data/pri_cust_asset_info.csv")[
        ["uid", "all_bal", "avg_mth", "avg_qur", "avg_year", "sa_bal", "td_bal", "fin_bal",
         "sa_crd_bal", "td_crd_bal", "sa_td_bal", "ntc_bal", "td_3m_bal", "td_6m_bal",
         "td_1y_bal", "td_2y_bal", "td_3y_bal", "td_5y_bal", "oth_td_bal", "cd_bal"]
    ].drop_duplicates(subset=["uid"], keep="first")
    cust_asset.set_index("uid", inplace=True)
    print(cust_asset)
    df = df.join(cust_asset, on="uid", how="left", sort=True)
    print(df)

    cust_asset_acct = pd.read_csv("../data/csv_data/pri_cust_asset_acct_info.csv",
                                  dtype={'avg_mth': 'float', 'avg_qur': 'float', 'avg_year': 'float'})[
        ["uid", "term", "deps_type", "is_secu_card", "acct_sts", "frz_sts", "stp_sts", "acct_bal", "bal", "avg_mth",
         "avg_qur", "avg_year"]
    ]
    cust_asset_acct.set_index("uid", inplace=True)
    print(cust_asset_acct)
    df = df.join(cust_asset_acct.drop_duplicates(subset=["uid"], keep="first"), on="uid", how="left", sort=True, rsuffix="cust_asset_acct")
    print(df)

    djk = pd.read_csv("../data/csv_data/dm_v_as_djk_info.csv", )[
        ["uid", "card_sts", "is_withdrw", "is_transfer", "is_deposit", "is_purchse", "cred_limit", "bankacct_bal",
         "is_mob_bank", "is_etc"]
    ]
    djk.set_index("uid", inplace=True)
    print(djk)
    df = df.join(djk.drop_duplicates(subset=["uid"], keep="first"), on="uid", how="left", sort=True, rsuffix="djk")
    print(df)

    df.to_csv("../data/test.csv", mode='w', line_terminator='\n',
              encoding='utf-8')


if __name__ == '__main__':
    # divide_train_test("credit")
    append_star_other_info("../data/processed_data/star_train.csv")
