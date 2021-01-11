from pprint import pprint
import warnings

from decision_system import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def read_excel_file(file_path):
    data_file = pd.read_excel(file_path)
    return data_file


def intro_menu():
    print("-----------------------------------------------------------------")
    print("      Welcome to the decision support system for customers")
    print("-----------------------------------------------------------------")


def initialize_app():
    intro_menu()
    file_path = Path(input("Give me the absolute path of the file you want to import:"))
    while not file_path.exists():
        file_path = Path(input("Wrong file path! Give me the correct absolute path of the file you want to import:"))

    return file_path


def import_settings():
    lindo_path = Path(input("Import the file with the lindo produced variables:"))
    while not lindo_path.exists():
        lindo_path = Path(input("File not exists!Give me a correct one:"))
    return lindo_path


if __name__ == '__main__':
    path = initialize_app()
    # System introduces the minimum sum error problem for decision systems
    my_system = DecisionSystem(path)
    # Find optimal distance indexes
    print("Equal distances for each characteristic: ", my_system.distance_index)

    # Export lindo settings and import the solutions of solver
    input("Press any key to continue...")
    my_system.export_lindo_data()
    print("Lindo Settings exported successfully to json file!")
    my_system.import_lindo_settings(import_settings())
    my_system.export_in_excel("Initial_Usage_List.xlsx")
    my_system.create_figures("Initial_Results.svg", "Initial Usage values for each Customer and Product")

    # Export lindo optimization settings and import the solutions of solver
    print("Optimazition step will be produced next.Loading data..")
    input("Press any key to continue...")
    while True:
        try:
            val = float(input("Enter the optimization step value:"))
            break
        except ValueError:
            print("Wrong user input. Try again!")
    my_system.export_optimizer_data(val)
    print("Lindo Settings exported successfully to json file!")
    my_system.import_lindo_settings(import_settings())
    my_system.export_in_excel("Optimizer_Usage_List.xlsx")
    my_system.create_figures("Optimization_Results.svg", "Optimization Usage values for each Customer and Product")
