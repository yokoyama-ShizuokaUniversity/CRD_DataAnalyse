import os
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
    }
    
    def __init__(
            self,
            dirpath: str,
            filename: str,
            debug: bool = False
        ):
        self.DIR_PATH = dirpath
        self.debug = debug
        self.filename = filename
        self._load_csv(filename)

    def _load_csv(self, filename: str) -> None:
        """CSVを読み込んでデータフレームとグループ分割を準備"""
        self.data_df = pd.read_csv(f"{self.DIR_PATH}/{filename}")
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

class GraphPlot:
    def __init__(self, data_loader: GetData, savefig: bool = False, debug: bool = False):
        self.data_loader = data_loader
        self.debug = debug
        self.dir_path = data_loader.DIR_PATH
        self.savefig = savefig

    def _plot(
            self,
            ax: Axes,
            title: str = None,
            ylabel: str = None,
            ylim: tuple[float, float] = None,
            yticks: list = None,
            savefig: bool = False,
            figtitle: str = None,
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
            ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if yticks:
            ax.set_yticks(yticks)
        if legend:
            ax.legend()
        plt.tight_layout()

        if self.savefig:
            try:
                plt.savefig(f"{savefig_path}/{title.replace(' ', '_')}")
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
            ax=ax,
            title="Success or Failed Percentage"
        )

if __name__ == "__main__":
    g_normal = GetData(
        dirpath="./",
        filename="all_apps_wide-2025-07-24.csv"
    )

    g_private = GetData(
        dirpath="./",
        filename="all_apps_wide-2025-08-07.csv",
    )

    for g in [g_normal, g_private]:
        # いろいろ情報を返す
        print(g)
        
        gp = GraphPlot(g, savefig=True)
        gp.plot_total_contribution()
        gp.plot_total_contribution(pledges=True)
        gp.plot_contribution_per_round()
        gp.plot_success_group_percentage()