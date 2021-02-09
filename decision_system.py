import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Optimizer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp


def gurobi_oprimizer(check):
    if check:
        return
    else:
        try:
            # Create a new model
            m = gp.Model("CustomerDecision")

            # Create variables
            s = np.empty(42, dtype=object)
            for index, element in enumerate(s):
                s[index] = m.addVar(vtype=GRB.BINARY, name="s" + str(index + 1))

            w11 = m.addVar(vtype=GRB.BINARY, name="w11")
            w12 = m.addVar(vtype=GRB.BINARY, name="w12")
            w13 = m.addVar(vtype=GRB.BINARY, name="w13")
            w21 = m.addVar(vtype=GRB.BINARY, name="w21")
            w22 = m.addVar(vtype=GRB.BINARY, name="w22")
            w23 = m.addVar(vtype=GRB.BINARY, name="w23")
            w31 = m.addVar(vtype=GRB.BINARY, name="w31")
            w32 = m.addVar(vtype=GRB.BINARY, name="w32")
            w41 = m.addVar(vtype=GRB.BINARY, name="w41")
            w42 = m.addVar(vtype=GRB.BINARY, name="w42")
            w51 = m.addVar(vtype=GRB.BINARY, name="w51")
            w52 = m.addVar(vtype=GRB.BINARY, name="w52")
            w53 = m.addVar(vtype=GRB.BINARY, name="w53")
            w61 = m.addVar(vtype=GRB.BINARY, name="w61")
            w62 = m.addVar(vtype=GRB.BINARY, name="w62")
            w63 = m.addVar(vtype=GRB.BINARY, name="w63")
            w64 = m.addVar(vtype=GRB.BINARY, name="w64")

            # Set objective
            m.setObjective(sum(s), GRB.MINIMIZE)

            # Add constraints
            # DM1
            m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 -
                        (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0) - s[0] + s[1] + s[2] - s[3] >= 0.05, "c0")
            m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0 -
                        (0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61) -
                        s[2] + s[3] + s[4] - s[5] >= 0.05, "c1")
            m.addConstr(0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61 -
                        (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[4] + s[
                            5] + s[6] - s[7] >= 0.05, "c2")
            m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 -
                        (
                                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64) -
                        s[6] + s[7] + s[8] - s[9] >= 0.05, "c3")
            m.addConstr(
                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64 -
                (
                        w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) -
                s[8] + s[9] + s[10] - s[11] >= 0.05, "c4")
            m.addConstr(
                w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 -
                (
                        0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                s[10] + s[11] + s[12] - s[13] == 0.0, "c5")

            # DM2
            m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 1.0 * w61 -
                        (
                                0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64) -
                        s[14] + s[15] + s[16] - s[17] >= 0.05, "c6")
            m.addConstr(
                0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 -
                (
                        w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62) -
                s[16] + s[17] + s[18] - s[19] == 0.0, "c7")
            m.addConstr(
                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62 -
                (
                        0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                s[18] + s[19] + s[20] - s[21] >= 0.05, "c8")
            m.addConstr(
                0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 -
                (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0) - s[20] + s[
                    21] + s[22] - s[23] >= 0.05, "c9")
            m.addConstr(w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0 -
                        (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) - s[22] + s[23] + s[24] -
                        s[25] == 0.0, "c10")
            m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 -
                        (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) -
                        s[24] + s[25] + s[26] - s[27] >= 0.05, "c11")

            # DM3
            m.addConstr(
                0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 -
                (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[28] + s[29] + s[
                    30] - s[31] >= 0.05, "c12")
            m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 -
                        (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) -
                        s[30] + s[31] + s[32] - s[33] >= 0.05, "c13")
            m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 -
                        (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61) - s[32] + s[33] + s[34] - s[
                            35] == 0.0, "c14")
            m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61 -
                        (
                                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63) -
                        s[34] + s[35] + s[36] - s[37] >= 0.05, "c15")
            m.addConstr(
                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63 -
                (
                        0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                s[36] + s[37] + s[38] - s[39] >= 0.05, "c16")
            m.addConstr(
                0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 -
                (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 1.0 * w61) - s[38] +
                s[39] + s[40] - s[41] >= 0.05, "c17")

            # S and W constraints
            for index, element in enumerate(s):
                m.addConstr(s + (index + 1) >= 0.0, "c" + str(index + 18))

            m.addConstr(w11 >= 0.0, "c60")
            m.addConstr(w12 >= 0.0, "c61")
            m.addConstr(w13 >= 0.0, "c62")
            m.addConstr(w21 >= 0.0, "c63")
            m.addConstr(w22 >= 0.0, "c64")
            m.addConstr(w23 >= 0.0, "c65")
            m.addConstr(w31 >= 0.0, "c66")
            m.addConstr(w32 >= 0.0, "c67")
            m.addConstr(w41 >= 0.0, "c68")
            m.addConstr(w42 >= 0.0, "c69")
            m.addConstr(w51 >= 0.0, "c70")
            m.addConstr(w52 >= 0.0, "c71")
            m.addConstr(w53 >= 0.0, "c72")
            m.addConstr(w61 >= 0.0, "c73")
            m.addConstr(w62 >= 0.0, "c74")
            m.addConstr(w63 >= 0.0, "c75")
            m.addConstr(w64 >= 0.0, "c76")

            # Optimize model
            m.optimize()

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')


class DecisionSystem:
    def __init__(self, file):
        self.df = self._load(file)
        self.order_data = self.df.loc["Data Order", :]
        self.df = self.df.iloc[:-1]
        self.total_chars = self.get_total_characteristics()
        self.total_customers = self.get_total_customers()
        self.total_options = self.get_options()
        self.max_spaces = 5
        self.s_count = 1
        self.distance_index = self.find_distance_indexes()
        self.weights_per_option = self.calculate_weights()
        self.temp_weights = self.calculate_weights()
        self.weights_per_ranking = self.calculate_ranking_weights()
        self.options_usage = self.create_usage_list()
        self.imported_data = {}
        self.lindo_F = 0.0

    @staticmethod
    def _load(data):
        if isinstance(data, (str, Path)):
            return pd.read_excel(data, header=1).iloc[:-1].set_index("Car")
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise TypeError(f"Unsupported init type: {type(data)}")

    # Initital methods processing data
    def get_value(self, row, col):
        return self.df.loc[row, col]

    def get_total_characteristics(self):
        return self.df.columns.get_loc("Ranking1")  # Format of excel is specific

    def get_total_customers(self):
        # Calculated by TotalNumberColumns - (totalChars + 1) / 2
        # totalChars + 1 = Because index starts from zero
        # /2 =Because every customer sets 2 opinions for design and ranking
        return int((len(self.df.columns) - (self.total_chars - 1)) / 2)

    def get_options(self):
        return len(self.df.index)

    def find_distance_indexes(self):
        index_dictionary = {}

        for idx, val in enumerate(self.df.columns):
            if "Ranking" in val:
                continue
            elif "Design" in val:
                if "Design" in index_dictionary:
                    continue
                index_dictionary["Design"] = self.calculate_point_distances(val)
            else:
                index_dictionary[val] = self.calculate_point_distances(val)
                if "Acceleration" in val:
                    index_dictionary[val] = index_dictionary[val] - 1

        return index_dictionary

    def create_usage_list(self):
        usage_dict = {}
        for x in range(self.total_customers):
            usage_dict["DM" + str(x + 1)] = {}
            for idx, y in enumerate(self.df.index):
                usage_dict["DM" + str(x + 1)][y] = 0.0

        return usage_dict

    # Calculation methods
    # Ipologismos twn diastimatwn twn varwn gia tis epiloges mas
    # Vasizetai  sto diastima timwn kai tis diathesimes epiloges
    def calculate_point_distances(self, col_name):
        max_value = max(self.df[col_name])
        min_value = min(self.df[col_name])
        equal_spaces = 1
        range_values = round(max_value - min_value)

        if range_values < (self.get_options() - 2):
            if range_values >= self.max_spaces:
                equal_spaces = self.max_spaces
            else:
                equal_spaces = range_values
        else:
            i = self.max_spaces
            while i > 1:
                if (range_values % i) == 0:
                    equal_spaces = i
                    if i == 3:
                        break
                i = i - 1

        return equal_spaces

    # Calculate the function of each customer of each option depending on characteristics
    # Ipologismos tis sinolikis xrisimotitas gia kathe katanalwti gia kathe epilogi gia kathe xaraktiristiko
    def calculate_weights(self):
        option_weights = {}

        for x in range(self.total_customers):
            option_weights["DM" + str(x + 1)] = {}

            for index, row in enumerate(self.df.index):
                option_weights["DM" + str(x + 1)][row] = ""
                for idx, col in enumerate(self.df.columns):
                    if "Design1" in col:
                        col = "Design" + str(x + 1)
                    elif "Design" in col:
                        continue

                    if "Ranking" in col:
                        continue

                    # Ipologismos diastimatwn me vasi tin seira katata3is
                    if self.order_data[col] == "INCR":
                        max_value = max(self.df[col])
                        min_value = min(self.df[col])
                    else:
                        max_value = min(self.df[col])
                        min_value = max(self.df[col])
                    range_values = abs(max_value - min_value)

                    if "Design" in col:
                        spacing = range_values / self.distance_index["Design"]
                    else:
                        spacing = range_values / self.distance_index[col]

                    if self.order_data[col] == "DECR":
                        spacing = -spacing

                    current_value = self.get_value(row, col)
                    temp_value = min_value
                    count = 1

                    if current_value == min_value:
                        if "Design" in col:
                            option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                row) + str(0.0)
                        continue
                    else:
                        while True:
                            if ((spacing + temp_value) >= current_value and self.order_data[col] == "INCR") or \
                                    ((spacing + temp_value) <= current_value and self.order_data[col] == "DECR"):
                                if "Design" in col:
                                    option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                        row) + str(
                                        round(abs((current_value - temp_value) / spacing), 2)) + " * w" + str(
                                        idx + 1) + str(count)
                                else:
                                    option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                        row) + str(
                                        round(abs((current_value - temp_value) / spacing), 2)) + " * w" + str(
                                        idx + 1) + str(count) + " + "
                                break
                            else:
                                option_weights["DM" + str(x + 1)][row] = option_weights["DM" + str(x + 1)].get(
                                    row) + "w" + str(
                                    idx + 1) + str(count) + " + "
                                count += 1
                                temp_value += spacing

        return option_weights

    # Calculate the function of each customer of each option depending on ranking
    def calculate_ranking_weights(self):
        ranking_weights = {}
        # Make type integer of ranking so that it was be sorted
        for idx, col in enumerate(self.df.columns):
            if "Ranking" in col:
                self.df[col] = self.df[col].astype(int)

        for x in range(self.total_customers):
            ranking_weights["DM" + str(x + 1)] = {}
            temp_df = self.df.sort_values(by=("Ranking" + str(x + 1)), ascending=True)
            current_col_index = temp_df.columns.get_loc("Ranking" + str(x + 1))
            for index, row in enumerate(temp_df.index):
                if temp_df.iloc[index, current_col_index] == temp_df.iloc[index + 1, current_col_index]:
                    ranking_weights["DM" + str(x + 1)]["D(" + row + "-" + temp_df.index[index + 1]] = \
                        self.weights_per_option["DM" + str(x + 1)][row] + " - (" + \
                        self.weights_per_option["DM" + str(x + 1)][temp_df.index[index + 1]] + ") - s" + str(
                            self.s_count) + " + s" + str(self.s_count + 1) + " + s" + str(
                            self.s_count + 2) + " - s" + str(
                            self.s_count + 3) + " = 0.0"
                else:  # Greater only because it is sorted
                    ranking_weights["DM" + str(x + 1)]["D(" + row + "-" + temp_df.index[index + 1]] = \
                        self.weights_per_option["DM" + str(x + 1)][
                            row] + " - (" + self.weights_per_option["DM" + str(x + 1)][
                            temp_df.index[index + 1]] + ") - s" + str(self.s_count) + " + s" + str(
                            self.s_count + 1) + " + s" + str(self.s_count + 2) + " - s" + str(
                            self.s_count + 3) + " >= 0.05"
                self.s_count += 2
                if index == self.get_options() - 2:
                    break
            self.s_count += 2

        return ranking_weights

    # Epe3ergasia dedomenwn eisodou apo to Lindo - Antikatastasi varwn stis times xrisimotitas twm autokinitwn gia kathe katanalwti
    def process_import_data(self):
        temp_weights_option = self.weights_per_option
        if "F" in self.imported_data:
            self.lindo_F = self.imported_data["F"]
        for x in range(self.total_customers):
            for index, row in enumerate(self.df.index):
                for idx, val in enumerate(self.imported_data["Weights"]):
                    for i, k_val in enumerate(self.imported_data["Weights"][val]):
                        if k_val in temp_weights_option["DM" + str(x + 1)][row]:
                            temp_value = temp_weights_option["DM" + str(x + 1)][row].replace(k_val, self.imported_data[
                                "Weights"][val][k_val])
                            temp_weights_option["DM" + str(x + 1)][row] = temp_value
                self.options_usage["DM" + str(x + 1)][row] = eval(temp_weights_option["DM" + str(x + 1)][row])
        self.imported_data = {}
        self.weights_per_option = self.temp_weights
        return

    # Import-Export data
    def import_lindo_settings(self, file):
        with open(file) as f:
            data = json.load(f)
        self.imported_data = data
        self.process_import_data()
        return

    # Eksagwgi twn dedomenwn eisodou sto lindo stin arxiki periptwsi
    def export_lindo_data(self):
        temp_dict = self.weights_per_ranking
        temp_dict["F"] = "min{Sum(si) from i=1 to " + str(self.s_count - 1) + "}"
        temp_dict[
            "Conditions"] = "Sum(wij) = 1, s(i), w(ij) >= 0 where i = " + str(
            self.get_total_characteristics()) + " and j = distance for each characteristic"
        with open('Initial_Lindo_Settings.json', 'w') as outfile:
            json.dump(temp_dict, outfile)  # Dimiourgia JSON arxeiou
        return

    # Eksagwgi twn dedomenwn eisodou sto lindo sto provlima tis veltistopoihshs
    def export_optimizer_data(self, e_val):
        temp_dict = self.weights_per_ranking
        temp_dict["F"] = {}
        for idx, y in enumerate(self.distance_index):
            temp_str = ""
            for x in range(self.distance_index[y]):
                if x == 0:
                    temp_str = "w" + str(idx + 1) + str(x + 1)
                else:
                    temp_str = temp_str + " + " + "w" + str(idx + 1) + str(x + 1)
            temp_dict["F"][str(idx + 1)] = "max{" + temp_str + "}"
            temp_dict["Condition1"] = "Sum(wij) = 1, s(i), w(ij) >= 0 where i = " + str(
                self.get_total_characteristics()) + " and j = distance for each characteristic"
            lindo_val = (1.0 + e_val) * float(self.lindo_F)
            temp_dict["Condition2"] = "Sum(si)<=" + str(lindo_val) + " from i=1 to " + str(
                self.s_count - 1)
            with open('Optimization_Lindo_Settings.json', 'w') as outfile:
                json.dump(temp_dict, outfile)
        return

    def export_in_excel(self, filename):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df = pd.DataFrame(self.options_usage.values(),
                              index=self.options_usage.keys())
            df.to_excel(writer, sheet_name='Usage List')
        print('Excel file with usage list created Successfully')
        return

    def create_figures(self, filename, set_title):
        data = pd.DataFrame(self.options_usage.values(), index=self.options_usage.keys())
        data.plot(kind='bar', figsize=(14, 9), xlabel='Customers', ylabel='Usage Value',
                  title=set_title)
        plt.legend(frameon=False, loc='upper center', ncol=7)
        plt.savefig(filename)
        plt.show()

    def gurobi_optimizer(self, check):
        if check:
            return
        else:
            try:

                # Create a new model
                m = gp.Model("CustomerDecision")

                # Create variables
                s = np.empty(42, dtype=object)
                for index, element in enumerate(s):
                    s[index] = m.addVar(vtype=GRB.BINARY, name="s" + str(index + 1))

                w11 = m.addVar(vtype=GRB.BINARY, name="w11")
                w12 = m.addVar(vtype=GRB.BINARY, name="w12")
                w13 = m.addVar(vtype=GRB.BINARY, name="w13")
                w21 = m.addVar(vtype=GRB.BINARY, name="w21")
                w22 = m.addVar(vtype=GRB.BINARY, name="w22")
                w23 = m.addVar(vtype=GRB.BINARY, name="w23")
                w31 = m.addVar(vtype=GRB.BINARY, name="w31")
                w32 = m.addVar(vtype=GRB.BINARY, name="w32")
                w41 = m.addVar(vtype=GRB.BINARY, name="w41")
                w42 = m.addVar(vtype=GRB.BINARY, name="w42")
                w51 = m.addVar(vtype=GRB.BINARY, name="w51")
                w52 = m.addVar(vtype=GRB.BINARY, name="w52")
                w53 = m.addVar(vtype=GRB.BINARY, name="w53")
                w61 = m.addVar(vtype=GRB.BINARY, name="w61")
                w62 = m.addVar(vtype=GRB.BINARY, name="w62")
                w63 = m.addVar(vtype=GRB.BINARY, name="w63")
                w64 = m.addVar(vtype=GRB.BINARY, name="w64")

                # Set objective
                m.setObjective(sum(s), GRB.MINIMIZE)

                # Add constraints
                # DM1
                m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 -
                            (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0) - s[0] + s[1] + s[2] - s[3] >= 0.05, "c0")
                m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0 -
                            (0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61) -
                            s[2] + s[3] + s[4] - s[5] >= 0.05, "c1")
                m.addConstr(0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61 -
                            (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[4] + s[
                                5] + s[6] - s[7] >= 0.05, "c2")
                m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 -
                            (
                                        w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64) -
                            s[6] + s[7] + s[8] - s[9] >= 0.05, "c3")
                m.addConstr(
                    w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64 -
                    (
                                w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) -
                    s[8] + s[9] + s[10] - s[11] >= 0.05, "c4")
                m.addConstr(
                    w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 -
                    (
                                0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                    s[10] + s[11] + s[12] - s[13] == 0.0, "c5")

                # DM2
                m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 1.0 * w61 -
                            (
                                        0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64) -
                            s[14] + s[15] + s[16] - s[17] >= 0.05, "c6")
                m.addConstr(
                    0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 -
                    (
                                w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62) -
                    s[16] + s[17] + s[18] - s[19] == 0.0, "c7")
                m.addConstr(
                    w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62 -
                    (
                                0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                    s[18] + s[19] + s[20] - s[21] >= 0.05, "c8")
                m.addConstr(
                    0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 -
                    (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0) - s[20] + s[
                        21] + s[22] - s[23] >= 0.05, "c9")
                m.addConstr(w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0 -
                            (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) - s[22] + s[23] + s[24] -
                            s[25] == 0.0, "c10")
                m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 -
                            (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) -
                            s[24] + s[25] + s[26] - s[27] >= 0.05, "c11")

                # DM3
                m.addConstr(
                    0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 -
                    (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[28] + s[29] + s[
                        30] - s[31] >= 0.05, "c12")
                m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 -
                            (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) -
                            s[30] + s[31] + s[32] - s[33] >= 0.05, "c13")
                m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 -
                            (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61) - s[32] + s[33] + s[34] - s[
                                35] == 0.0, "c14")
                m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61 -
                            (
                                        w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63) -
                            s[34] + s[35] + s[36] - s[37] >= 0.05, "c15")
                m.addConstr(
                    w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63 -
                    (
                                0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) -
                    s[36] + s[37] + s[38] - s[39] >= 0.05, "c16")
                m.addConstr(
                    0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 -
                    (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 1.0 * w61) - s[38] +
                    s[39] + s[40] - s[41] >= 0.05, "c17")

                # S and W constraints
                for index, element in enumerate(s):
                    m.addConstr(s + (index + 1) >= 0.0, "c" + str(index + 18))

                m.addConstr(w11 >= 0.0, "c60")
                m.addConstr(w12 >= 0.0, "c61")
                m.addConstr(w13 >= 0.0, "c62")
                m.addConstr(w21 >= 0.0, "c63")
                m.addConstr(w22 >= 0.0, "c64")
                m.addConstr(w23 >= 0.0, "c65")
                m.addConstr(w31 >= 0.0, "c66")
                m.addConstr(w32 >= 0.0, "c67")
                m.addConstr(w41 >= 0.0, "c68")
                m.addConstr(w42 >= 0.0, "c69")
                m.addConstr(w51 >= 0.0, "c70")
                m.addConstr(w52 >= 0.0, "c71")
                m.addConstr(w53 >= 0.0, "c72")
                m.addConstr(w61 >= 0.0, "c73")
                m.addConstr(w62 >= 0.0, "c74")
                m.addConstr(w63 >= 0.0, "c75")
                m.addConstr(w64 >= 0.0, "c76")

                # Optimize model
                m.optimize()

            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))

            except AttributeError:
                print('Encountered an attribute error')
            return

