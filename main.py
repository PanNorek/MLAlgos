from ArgHelper import arg_helper
import os

if __name__ == '__main__':
    args = arg_helper()
    print(args)
    selected_model = args['model']
    data_path = args['data']
    
    os.system(f"python ./src/models/{selected_model}.py ../{data_path}")
    
    



