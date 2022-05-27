import math
import time

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
        train.to_csv("../data/processed_data/l_star_train.csv", index=False, mode='w', line_terminator='\n',
                     encoding='utf-8')
        test.to_csv("../data/processed_data/l_star_test.csv", index=False, mode='w', line_terminator='\n',
                    encoding='utf-8')
    elif info_type == "credit":
        df = pd.read_csv("../data/csv_data/pri_credit_info.csv").fillna("-1")
        train = df[(df["credit_level"] != -1)]
        test = df[(df["credit_level"] == -1)]
        print(train)
        print(test)
        train.to_csv("../data/processed_data/l_credit_train.csv", index=False, mode='w', line_terminator='\n',
                     encoding='utf-8')
        test.to_csv("../data/processed_data/l_credit_test.csv", index=False, mode='w', line_terminator='\n',
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
    ]
    print(cust_asset)
    cust_asset = cust_asset.drop_duplicates(subset=["uid"], keep="first")
    cust_asset.set_index("uid", inplace=True)
    df = df.join(cust_asset, on="uid", how="left", sort=True)
    print(df)

    cust_asset_acct = pd.read_csv("../data/csv_data/pri_cust_asset_acct_info.csv",
                                  dtype={'avg_mth': 'float', 'avg_qur': 'float', 'avg_year': 'float'})[
        ["uid", "term", "deps_type", "is_secu_card", "acct_sts", "frz_sts", "stp_sts", "acct_bal", "bal", "avg_mth",
         "avg_qur", "avg_year"]
    ]
    print(cust_asset_acct)
    cust_asset_acct = cust_asset_acct.drop_duplicates(subset=["uid"], keep="first")
    cust_asset_acct.set_index("uid", inplace=True)
    df = df.join(cust_asset_acct, on="uid", how="left", sort=True,
                 rsuffix="cust_asset_acct")
    print(df)

    djk = pd.read_csv("../data/csv_data/dm_v_as_djk_info.csv", )[
        ["uid", "card_sts", "is_withdrw", "is_transfer", "is_deposit", "is_purchse", "cred_limit", "bankacct_bal",
         "is_mob_bank", "is_etc"]
    ]
    print(djk)
    djk = djk.drop_duplicates(subset=["uid"], keep="first")
    djk.set_index("uid", inplace=True)
    df = df.join(djk, on="uid", how="left", sort=True, rsuffix="djk")
    print(df)

    djkfq = pd.read_csv("../data/csv_data/dm_v_as_djkfq_info.csv", )[
        ["uid", "mp_type", "mp_status"]
    ]
    print(djkfq)
    djkfq = djkfq.drop_duplicates(subset=["uid"], keep="first")
    djkfq.set_index("uid", inplace=True)
    df = df.join(djkfq, on="uid", how="left", sort=True, rsuffix="djkfq")
    print(df)

    df.to_csv("../data/processed_data/star_test.csv", mode='w', line_terminator='\n',
              encoding='utf-8')


def append_credit_other_info(credit_path):
    df = pd.read_csv(credit_path)
    df.set_index("uid", inplace=True)
    print(df)
    cust_liab_acct = pd.read_csv("../data/csv_data/processed_cust_liab_acct_info.csv")
    cust_liab_acct.set_index("uid", inplace=True)
    print(cust_liab_acct)
    df = df.join(cust_liab_acct, on="uid", how="left", sort=True, rsuffix="cust_liab_acct")
    print(df)

    cust_liab = pd.read_csv("../data/csv_data/pri_cust_liab_info.csv")[
        ["uid", "all_bal", "bad_bal", "due_intr", "norm_bal", "delay_bal"]
    ]
    print(cust_liab)
    cust_liab = cust_liab.drop_duplicates(subset=["uid"], keep="first")
    cust_liab.set_index("uid", inplace=True)
    print(cust_liab)
    df = df.join(cust_liab, on="uid", how="left", sort=True, rsuffix="cust_liab")
    print(df)

    djk = pd.read_csv("../data/csv_data/dm_v_as_djk_info.csv", )[
        ["uid", "card_sts", "is_withdrw", "is_transfer", "is_deposit", "is_purchse", "cred_limit", "over_draft",
         "dlay_amt", "bankacct_bal", "is_mob_bank", "is_etc", "dlay_mths"]
    ]
    print(djk)
    djk = djk.drop_duplicates(subset=["uid"], keep="first")
    djk.set_index("uid", inplace=True)
    df = df.join(djk, on="uid", how="left", sort=True, rsuffix="djk")
    print(df)

    djkfq = pd.read_csv("../data/csv_data/dm_v_as_djkfq_info.csv", )[
        ["uid", "mp_type", "mp_status", "total_amt", "total_mths",
         "mth_instl", "instl_cnt", "rem_ppl", "total_fee", "rem_fee"]
    ]
    print(djkfq)
    djkfq = djkfq.drop_duplicates(subset=["uid"], keep="first")
    djkfq.set_index("uid", inplace=True)
    df = df.join(djkfq, on="uid", how="left", sort=True, rsuffix="djkfq")
    print(df)
    df = df.fillna(0)
    df.to_csv("../data/processed_data/credit_train.csv", mode='w', line_terminator='\n',
              encoding='utf-8')


def credit_process():
    df = pd.read_csv("../data/csv_data/pri_cust_liab_acct_info.csv")[
        ["cust_name", "loan_amt", "loan_type", "loan_bal", "vouch_type", "is_mortgage",
         "is_online", "is_extend", "five_class", "overdue_class", "overdue_flag", "owed_int_flag", "credit_amt",
         "defect_type", "owed_int_in", "owed_int_out", "delay_bal", "acct_sts", "is_book_acct"]
    ]

    grouped_df = df.groupby("cust_name")
    uid = []
    avg_loan_amt = []
    vouch_type = []
    is_mortgage = []
    is_online = []
    is_extend = []
    five_class = []
    overdue_class = []
    overdue_flag = []
    owed_int_flag = []
    credit_amt = []
    defect_type = []
    owed_int_in = []
    owed_int_out = []
    delay_bal = []
    acct_sts = []
    is_book_acct = []
    for name, group in grouped_df:
        print(name)
        uid.append(name)
        avg_loan_amt.append(group["loan_amt"].sum() / len(group["loan_amt"]))
        vouch_type.append(group["vouch_type"].loc[group["vouch_type"].first_valid_index()])
        is_mortgage.append(group["is_mortgage"].loc[group["is_mortgage"].first_valid_index()])
        is_online.append(group["is_online"].loc[group["is_online"].first_valid_index()])
        is_extend.append(group["is_extend"].loc[group["is_extend"].first_valid_index()])
        five_class.append(group["five_class"].loc[group["five_class"].first_valid_index()])
        overdue_class.append(group["overdue_class"].loc[group["overdue_class"].first_valid_index()])
        overdue_flag.append(group["overdue_flag"].loc[group["overdue_flag"].first_valid_index()])
        owed_int_flag.append(group["owed_int_flag"].loc[group["owed_int_flag"].first_valid_index()])
        credit_amt.append((group["credit_amt"].sum() / len(group["credit_amt"])))
        index = group["defect_type"].first_valid_index()
        if index is None:
            defect_type.append(None)
        else:
            defect_type.append(group["defect_type"].loc[index])
        owed_int_in.append(group["owed_int_in"].sum() / len(group["owed_int_in"]))
        owed_int_out.append(group["owed_int_out"].sum() / len(group["owed_int_out"]))
        delay_bal.append(group["delay_bal"].sum() / len(group["delay_bal"]))
        acct_sts.append(group["acct_sts"].loc[group["acct_sts"].first_valid_index()])
        index = group["is_book_acct"].first_valid_index()
        if index is None:
            is_book_acct.append(None)
        else:
            is_book_acct.append(group["is_book_acct"].loc[index])
    res = pd.DataFrame(
        data={
            "uid": uid,
            "avg_loan_amt": avg_loan_amt,
            "vouch_type": vouch_type,
            "is_mortgage": is_mortgage,
            "is_online": is_online,
            "is_extend": is_extend,
            "five_class": five_class,
            "overdue_class": overdue_class,
            "overdue_flag": overdue_flag,
            "owed_int_flag": owed_int_flag,
            "credit_amt": credit_amt,
            "defect_type": defect_type,
            "owed_int_in": owed_int_in,
            "owed_int_out": owed_int_out,
            "delay_bal": delay_bal,
            "acct_sts": acct_sts,
            "is_book_acct": is_book_acct
        }
    )
    print(res)
    res.to_csv("../data/csv_data/processed_cust_liab_acct_info.csv", mode='w', line_terminator='\n',
               encoding='utf-8')
    return res


if __name__ == '__main__':
    # divide_train_test("star")
    # append_star_other_info("../data/processed_data/l_star_test.csv")
    append_credit_other_info("../data/processed_data/l_credit_train.csv")
    # credit_process()
    # df = pd.read_csv("../data/csv_data/pri_cust_liab_info.csv")["delay_bal"]
    # for x in df:
    #     print(x)
    # df = pd.read_csv("../data/pri_credit_info.csv", sep="\t")
    # print(df)
