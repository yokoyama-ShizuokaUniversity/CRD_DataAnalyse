import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import ast

# vscode補完用
from matplotlib.axes import Axes

class GetData:
    # data_dfのカラム名
    COLUMNS = {
        "group": 'crd_cost_information.1.group.id_in_subsession',
        "label": 'participant.label',
        "finish_reason": 'participant.reason_finished',
        "comp_status": 'participant.completion_status',
        "g_target": 'crd_cost_information.1.group.target',
        "r1_pledge": 'crd_cost_information.1.player.pledge',
        "r6_pledge": 'crd_cost_information.6.player.pledge',
        "id": "participant.id_in_session",
        "surveys": "crd_cost_information_survey"
    }
    
    def __init__(
            self,
            dirpath: str,
            filename: str,
            debug: bool = False
        ):
        self.dir_path = dirpath
        self.debug = debug
        self.filename = filename
        self._load_csv(filename)

        # 日付をファイル名から抽出
        self.date = filename.split('-')[1:]
        self.date[2] = self.date[2].split('.')[0]

    def _load_csv(self, filename: str) -> None:
        """CSVを読み込んでデータフレームとグループ分割を準備"""
        self.data_df = pd.read_csv(f"{self.dir_path}/{filename}")
        groups = self.data_df.groupby(self.COLUMNS["group"])[self.COLUMNS["label"]].apply(list)
        self.group, self.ungrouped = (
            {g: labels for g, labels in groups.items() if len(labels) == 4},
            {g: labels for g, labels in groups.items() if len(labels) != 4},
        )

        if self.debug:
            print("grouped: ", self.group)
            print("ungrouped: ", self.ungrouped)

    def get_ungrouped_reason(self) -> dict[int, str]:
        """
        グループに4人揃わなかった参加者(label)ごとに
        理由(reason_finished)を返す。
        """
        reasons = {}
        for v in self.ungrouped.values():
            for ul in v:
                reason = self.data_df[self.data_df[self.COLUMNS["label"]] == ul][self.COLUMNS["finish_reason"]].iloc[0]
                reasons[ul] = reason

        if self.debug:
            print(reasons)
        
        return reasons
    
    def pick_passed_comprehension_test(self) -> tuple[list[int], list[int]]:
        """
        実験を最後まで終了した 'success' と 確認テストは完了したがマッチングできなかった'ungrouped' の
        参加者ラベルを返す。
        """
        success_label = self.data_df[self.data_df[self.COLUMNS["comp_status"]] == 'success'][self.COLUMNS['label']].tolist()
        ungrouped_label = self.data_df[self.data_df[self.COLUMNS["comp_status"]] == 'ungrouped'][self.COLUMNS['label']].tolist()
        return success_label, ungrouped_label

    def get_contributions(self) -> dict[int, list[int]]:
        """
        ラベルに対する各ラウンドの寄付額を辞書で返す
        """
        contribute_columns = [f'crd_cost_information.{i}.player.contribution' for i in range(1, 11)]
        label = self.data_df[self.COLUMNS["label"]].tolist()
        print(label)
        success_contribution = {}
        for sl in label:
            row = self.data_df[self.data_df[self.COLUMNS["label"]] == sl][contribute_columns]
            success_contribution[sl] = row.iloc[0].tolist()
        if self.debug:
            print(success_contribution)
        return success_contribution

    def get_success_contribution(self) -> dict[int, list[int]]:
        """
        理解テストsuccess者の寄付額(10ラウンド分)をリストで返す。
        """
        contribute_columns = [f'crd_cost_information.{i}.player.contribution' for i in range(1, 11)]
        success_label, _ = self.pick_passed_comprehension_test()
        success_contribution = {}
        for sl in success_label:
            row = self.data_df[self.data_df[self.COLUMNS["label"]] == sl][contribute_columns]
            success_contribution[sl] = row.iloc[0].tolist()
        if self.debug:
            print(success_contribution)
        return success_contribution
    
    def set_contribution_by_group(self) -> dict[int, pd.DataFrame]:
        """
        グループごとに、参加者の寄付した額をラウンドごとにまとめたDataFrameを辞書として返す。
        （キーがグループ番号）
        """
        round_column = [f"Round {i}" for i in range(1, 11)]
        success_contribution = self.get_success_contribution()
        dfs = {}
        for g, labels in self.group.items():
            part_contrib = []
            part_index = []
            for l in labels:
                try:
                    part_contrib.append(success_contribution[l])
                    part_index.append(l)
                except KeyError as e:
                    print(f"{l}のキーが存在しません。（success_contributionに含まれていない->success_labelにいない->途中離脱？）")
            df = pd.DataFrame(part_contrib, columns=round_column, index=part_index)
            dfs[g] = df
            if self.debug:
                print(df)
        if self.debug:
            print(dfs)
        return dfs
    
    def get_group_target(self) -> dict[int, int]:
        """
        各グループの目標額を辞書形式で返す
        """
        group_target = {}
        for g in self.group.keys():
            group_target[g] = self.data_df[self.data_df[self.COLUMNS["group"]] == g][self.COLUMNS["g_target"]].iloc[0]
        if self.debug:
            print(group_target)
        return group_target

    def split_target_success_orNot(self) -> tuple[list[int], list[int]]:
        """
        目標額を達成したかどうかで分離する
        """
        target_success_group = []
        target_failed_group = []

        for g in self.group.keys():
            group_mem = self.data_df[self.data_df[self.COLUMNS['group']] == g].iloc[0]
            if group_mem['crd_cost_information.10.group.failed'] == 0:
                target_success_group.append(g)
            else:
                target_failed_group.append(g)
        if self.debug:
            print(target_success_group, target_failed_group)

        return target_success_group, target_failed_group

    def get_participant_pledges(self) -> dict[int, list[int]]:
        """
        各参加者のラウンド1,6での寄付提案額を返す
        labelをキーとして、[group, round1_pledge, round6_pledge]で返す
        """
        participant_pledges = {}
        for g in self.group.keys():
            # 1ラウンドと6ラウンドの提案額
            pledges_df = self.data_df[self.data_df[self.COLUMNS['group']] == g][[self.COLUMNS['label'], self.COLUMNS['r1_pledge'], self.COLUMNS['r6_pledge']]]

            for index, row in pledges_df.iterrows():
                participant_pledges[int(row[self.COLUMNS['label']])] = [g, row[self.COLUMNS['r1_pledge']], row[self.COLUMNS['r6_pledge']]]
        
        if self.debug:
            print(participant_pledges)
        
        return participant_pledges
    
    def get_dropout_timing(self) -> dict[int, int]:
        """
        途中離脱した人を特定し、その人のラベルと退出したラウンドを返す
        """
        dropout_df = self.data_df[self.data_df[self.COLUMNS["comp_status"]] == 'timeout_decision']
        if dropout_df.empty:
            return {-1: "離脱者がいません。"}

        dropout_timing = {}
        for i in range(1, 11):
            dropout_list = ast.literal_eval(dropout_df[f'crd_cost_information.{i}.group.players_dropout'].iloc[0])
            if len(dropout_list) != 0:
                for dropout in dropout_list:
                    if self.data_df[self.data_df[self.COLUMNS['id']] == dropout][self.COLUMNS['label']].iloc[0] in dropout_timing.keys():
                        continue
                    dropout_timing[self.data_df[self.data_df[self.COLUMNS['id']] == dropout][self.COLUMNS['label']].iloc[0]] = i
                    if self.debug:
                        print(self.data_df[self.data_df[self.COLUMNS['id']] == dropout][self.COLUMNS['label']].iloc[0], "：退出ラウンド -", i)
        return dropout_timing

    def get_surveys(self) -> pd.DataFrame:
        """
        ラベルに対応した、アンケート回答結果を返すdf
        """
        columns = self.data_df.columns.tolist()
        grep_survey_columns = [c for c in columns if self.COLUMNS["surveys"] in c]
        df = self.data_df[self.COLUMNS["label"]]
        for c in grep_survey_columns:
            df = pd.concat([df, self.data_df[c]], axis=1)
        if self.debug:
            print(df)
        return df

    def __str__(self):
        """ 簡単にデータを表示 """
        ungrouped_reason = []
        dropouts = []
        for l, r in self.get_ungrouped_reason().items():
            ungrouped_reason.append(f"{l}: {r}")
        for l, t in self.get_dropout_timing().items():
            if l == -1:
                dropouts.append(f"{t}")
                break
            dropouts.append(f"{l}: Round {t} で離脱")
        s = [
            f"4人ちゃんと集まったグループ数：{len(self.group)}",
            str(self.group),
            "",
            f"4人集まらなかったグループ数：{len(self.ungrouped)}",
            str(self.ungrouped),
            "",
            f"正規に終了しなかった参加者数：{len(self.get_ungrouped_reason())}",
            f"4人そろわなかったグループの参加者の失敗理由：",
            "\n".join(ungrouped_reason),
            "",
            "途中離脱(ドロップアウト)した人：",
            "\n".join(dropouts),
        ]
        return "\n".join(s)

class YahooTask:
    def __init__(self, task_filename: str, experiment_cls: GetData=None, keywords: dict = None):
        self.gd = experiment_cls
        self.task_df = pd.read_csv(f"{self.gd.dir_path}/{task_filename}", sep='\t', header=None, encoding='utf8')
        if keywords:
            self.keywords = keywords
        else:
            # 2025-07の実験にて使用したキーワード
            self.keywords = {
                "A7K3B": "Success",
                "X7R5T": "Ungrouped",
            }

    def _extract_yahootask_columns(self) -> pd.DataFrame:
        task_list_concat_df = pd.DataFrame()
        task_list_concat_df['participant.label'] = self.task_df[4].str[-7:]
        task_list_concat_df['participant.label'] = task_list_concat_df['participant.label'].str.extract(r'^l=(\d+)$').astype('int')
        task_list_concat_df['yahoo_id'] = self.task_df[8]
        task_list_concat_df['key_word'] = self.task_df[7]
        task_list_concat_df['completion_status'] = task_list_concat_df['key_word'].map(self.keywords).fillna('Failed')

        return task_list_concat_df[task_list_concat_df['key_word'].map(self.keywords).notna()]
    
    def _extract_experiment_columns(self) -> pd.DataFrame:
        pg_concat_df = pd.DataFrame()
        pg_concat_df['participant.label'] = self.gd.data_df['participant.label']
        pg_concat_df['participant.payoff'] = self.gd.data_df.iloc[:,12]
        pg_concat_df['PayPay_exchanged'] = pg_concat_df['participant.payoff'].apply(lambda x: x * 4 + 100) # ECUからPayPayへの変換式はここで指定
        pg_concat_df['PayPay_ceiled'] = pg_concat_df["PayPay_exchanged"].apply(lambda x: ((x+9) // 10)*10) # ホワイトリストのために1の位切り上げたリスト
        pg_concat_df = pg_concat_df.dropna()
        pg_concat_df = pg_concat_df.astype('int')

        return pg_concat_df
    
    def merge_result(self) -> pd.DataFrame:
        return pd.merge(
            self._extract_yahootask_columns(),
            self._extract_experiment_columns(),
            on='participant.label',
            how='left'
        )
    
    def create_whitelist(self) -> None:
        merged_df = self.merge_result()
        paypay_whitelist_df = merged_df[["yahoo_id", "PayPay_exchanged", "PayPay_ceiled"]]
        whitelist_dict = {}
        for _, row in paypay_whitelist_df.iterrows():
            whitelist_dict[row["PayPay_ceiled"]] = []
        for _, row in paypay_whitelist_df.iterrows():
            whitelist_dict[row["PayPay_ceiled"]] += [row["yahoo_id"]]
        os.makedirs(f"{self.gd.dir_path}/whitelists", exist_ok=True)
        for k in whitelist_dict.keys():
            yahooid_list = whitelist_dict[k]
            print(f'{k}: {len(yahooid_list)}')
            with open(f"{self.gd.dir_path}/whitelists/{"".join(self.gd.date)}_{k}pt.txt", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(yahooid_list)
        print(sorted(whitelist_dict.keys()))
    
    def show_point_histgram(self) -> None:
        merged_df = self.merge_result()
        paypay_payoff = merged_df["PayPay_exchanged"].tolist()
        paypay_task_num = set(paypay_payoff)
        len(paypay_task_num)
        plt.hist(paypay_payoff, bins=100)
        plt.title(f"{"".join(self.gd.date)} -  PayPay Point")
        plt.show()

class GraphPlot:
    def __init__(self, data_loader: GetData, savefig: bool = False, debug: bool = False, figtype: str = None):
        self.data_loader = data_loader
        self.debug = debug
        self.dir_path = self.data_loader.dir_path
        self.savefig = savefig
        allowed_figtype = ["eps", "png"]
        if figtype in allowed_figtype:
            self.figtype = figtype
        elif figtype is None:
            self.figtype = "png"
        else:
            raise Exception(f"figtype Error : '{figtype}' is not allowed figtype")

    def _plot(
            self,
            ax: Axes,
            title: str = None,
            ylabel: str = None,
            xlabel: str = None,
            ylim: tuple[float, float] = None,
            yticks: list = None,
            savefig: bool = False,
            figtitle: str = None,
            figname: str = None,
            legend: bool = None,
            ) -> None:
        """
        表示の共通処理、表示または保存するための内部メソッド
        """
        csv_date = "".join(self.data_loader.filename.replace(".csv", '').split('-')[1:])
        # save dir check'
        if figtitle:
            savefig_path = f"{self.dir_path}/{csv_date}_Figures/{figtitle}"
        else:
            savefig_path = f"{self.dir_path}/{csv_date}_Figures"
        os.makedirs(savefig_path, exist_ok=True)

        if title:
            figname = title
            ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if yticks:
            ax.set_yticks(yticks)
        if legend:
            ax.legend(fontsize=16, loc="upper left")
        plt.tight_layout()

        if figname is None:
            figname = "Untitled"

        if self.savefig:
            try:
                plt.savefig(f"{savefig_path}/{figname.replace(' ', '_')}.{self.figtype}")
            except Exception as e:
                print(f"plt.savefig : {e}")
        else:
            plt.show()
        plt.close()

    def plot_total_contribution(self, pledges: bool = False, savefig: bool = None) -> None:
        """
        グループごとに、貢献額(contribution)を累積棒グラフ、目標額(target)を直線にて表示する。
        オプション：pledges = True のとき、1,6ラウンドでの提案額の直線を追加する。
        """
        if savefig:
            self.savefig = savefig

        figtitle = "total_contribution"
        if pledges:
            figtitle = "total_contribution_include_pledges"
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            participant_pledges = self.data_loader.get_participant_pledges()

        dfs = self.data_loader.set_contribution_by_group()
        group_target = self.data_loader.get_group_target()
        for g, labels in self.data_loader.group.items():
            df = dfs[g].cumsum(axis=1)
            df = df.T

            ax = df.plot(kind='bar', stacked=True)
            ax.axhline(y=group_target[g], color='red', linestyle='--', label='Target')            
            ax.legend()

            if pledges:
                for i, l in enumerate(labels):
                    r1_pledge = participant_pledges[l][1]
                    r6_pledge = participant_pledges[l][2]
                    ax.plot([0, 5], [r1_pledge, r1_pledge], color=colors[i], linestyle='--', label=f'{l}_predges')
                    ax.plot([5, 9], [r6_pledge, r6_pledge], color=colors[i], linestyle='--', label=f'{l}_predges')                
            
            self._plot(
                ax=ax,
                title=f"Group {g}",
                ylabel="Total contribution (ECU)", 
                ylim=(0, 160),
                figtitle=figtitle,
            )
            if self.debug:
                break

    def plot_contribution_per_round(self, savefig: bool = False) -> None:
        """
        グループごとに、各ラウンドで、いくら貢献したかを折れ線グラフにて表示する。
        """
        if savefig:
            self.savefig = savefig
        
        dfs = self.data_loader.set_contribution_by_group()
        for g in self.data_loader.group:
            df = dfs[g]
            df = df.T

            ax = df.plot(kind='line', marker='o')
            self._plot(
                ax=ax, title=f"Group {g}",
                ylabel="Contribution per round (ECU)",
                yticks=[0, 2, 4],
                figtitle="contribution_per_round",
                legend=True,
            )
            if self.debug:
                break

    def plot_success_group_percentage(self, savefig: bool = False) -> None:
        """
        成功したグループの割合を棒グラフで表示する。
        """
        if savefig:
            self.savefig = savefig

        success_group, failed_group = self.data_loader.split_target_success_orNot()
        label = ["Success", "Failed"]
        success_or_not_group = [len(success_group), len(failed_group)]
        success_or_not_per_group = [x/sum(success_or_not_group) for x in success_or_not_group]

        fig, ax = plt.subplots()
        ax.bar(label, success_or_not_per_group, width=0.8)
        self._plot(
            ax=ax
        )

    def plot_success_percentage_ver2(self, other_datacls: GetData = None, savefig: bool = False):
        """
        9/3追記
        グループの割合を棒グラフで表示する。1枚にまとめ、成功率だけ表示
        """
        if savefig:
            self.savefig = savefig

        if other_datacls is None:
            raise Exception("比較する方のインスタンスを渡してください。")
        print(f"[Uncertainty] DataLoader : {self.data_loader.filename}")
        print(f"[Uncertainty + Info] DataLoader : {other_datacls.filename}")

        c_success_grp, c_failed_grp = self.data_loader.split_target_success_orNot()
        cpi_success_grp, cpi_failed_grp = other_datacls.split_target_success_orNot()

        c_success_rate = len(c_success_grp) / (len(c_success_grp) + len(c_failed_grp))
        cpi_success_rate = len(cpi_success_grp) / (len(cpi_success_grp) + len(cpi_failed_grp))

        label = ["Uncertainty", "Uncertainty + Info"]

        fig, ax = plt.subplots()
        ax.bar([0], [c_success_rate], width=0.8, facecolor="white", edgecolor="black", linewidth=2)
        ax.bar([1], [cpi_success_rate], width=0.8, facecolor="black", edgecolor="black", linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(label, fontsize=20)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Fraction of successful groups", fontsize=20)
        self._plot(
            ax=ax,
            ylim=(0, 0.8),
            figname="Success_percentage"
        )

    def plot_individual_contrib(self, savefig: bool = False, bartype: str = "white"):
        """
        9/3
        各参加者(individual)の貢献額
        """
        if savefig:
            self.savefig = savefig
        
        contrib_dict = self.data_loader.get_success_contribution()
        items = [(k, sum(v)) for k, v in contrib_dict.items()]
        sorted_contrib = sorted(items, key=lambda x: x[1])
        y = [v[1] for v in sorted_contrib]

        contrib_average = sum(y) / len(y)

        fig, ax = plt.subplots()
        ax.bar(range(len(y)), y, facecolor=bartype, edgecolor="black", label="Individual")
        ax.axhline(y=contrib_average, color="red", linestyle="--", linewidth=2, label="Average")
        ax.text(1, contrib_average+1, f"Average : {contrib_average:.3f}", color="red", fontsize=20)
        ax.tick_params(axis="x", length=0)
        ax.set_xticklabels("")
        ax.set_ylabel("Individual total contributions (ECUs)", fontsize=16)
        ax.set_xlabel("Individual (sorted)", fontsize=20)
        figname = "individual_contrib" if bartype == "white" else "individual_contrib_+info"
        self._plot(
            ax=ax,
            ylim=(0, 42),
            legend=True,
            figname=figname
        )
    
    def plot_group_contrib(self, savefig: bool = False, bartype: str = "white"):
        contrib_dict = self.data_loader.get_contributions()
        group_target = self.data_loader.get_group_target()
        print(contrib_dict)
        group_contrib = []
        for k, v in self.data_loader.group.items():
            group_c = 0
            for l in v:
                group_c += sum(contrib_dict[l])
            group_contrib.append((k, group_c))
        sorted_contrib = sorted(group_contrib, key=lambda x: x[1])
        y = [v[1] for v in sorted_contrib]
        target_y = [group_target[v[0]] for v in sorted_contrib]
        contrib_average = sum(y) / len(y)

        fig, ax = plt.subplots()
        ax.bar(range(len(y)), y, facecolor=bartype, edgecolor="black", label="Group")
        ax.scatter(range(len(target_y)), target_y, marker="o", color="orange", label="Target")
        ax.axhline(y=contrib_average, color="red", linestyle="--", linewidth=2, label="Average")
        if bartype == "white":
            ax.text(12, contrib_average-11, f"Average : {contrib_average:.1f}", color="red", fontsize=20)
        else:
            ax.text(0, contrib_average+1.5, f"Average : {contrib_average:.1f}", color="red", fontsize=20)

        ax.set_xlabel("Group (sorted)", fontsize=20)
        ax.set_ylabel("Final public account (ECUs)", fontsize=20)
        ax.tick_params(axis="x", length=0)
        ax.set_xticklabels("")
        figname = "group_contrib" if bartype == "white" else "group_contrib_+info"
        self._plot(
            ax=ax,
            ylim=(0, 165),
            figname=figname,
            legend=True,
        )

if __name__ == "__main__":
    """ Usage - 使い方 """
    
    # yahooのtsvと実験データcsvセットを要確認！
    g_normal = GetData(
        dirpath="./",
        filename="all_apps_wide-2025-07-24.csv"
    )
    ytask_normal = YahooTask("3589457371.tsv", g_normal)

    # 簡単な実験から得られたデータを表示
    print(g_normal)

    # ホワイトリストを作成、成果報酬のヒストグラムを表示
    ytask_normal.create_whitelist()
    ytask_normal.show_point_histgram()

    # グラフを作成・表示・保存
    gp = GraphPlot(g_normal, savefig=True, debug=True)
    gp.plot_total_contribution()
    gp.plot_total_contribution(pledges=True)
    gp.plot_contribution_per_round()
    gp.plot_success_group_percentage()