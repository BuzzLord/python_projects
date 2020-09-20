
import os
import sys
import json
from siren06 import main

if __name__ == "__main__":
    file_list = sys.argv[1:]
    successful = []
    for f in file_list:
        try:
            if not os.path.exists(f):
                print("Path {} does not exist".format(f))
                continue
            with open(f, 'r') as fp:
                json_args = json.load(fp)
            if isinstance(json_args, dict):
                args = []
                for k in json_args.keys():
                    args.append("--{}".format(k))
                    if isinstance(json_args[k], str) and len(json_args[k]) == 0:
                        continue
                    elif isinstance(json_args[k], list):
                        args.extend([str(i) for i in json_args[k] if len(str(i)) > 0])
                    else:
                        args.append(str(json_args[k]))
            elif isinstance(json_args, list):
                args = [str(i) for i in json_args]
            else:
                print("Json file {} should be dict or list, not {}".format(f, type(json_args)))
                continue

            print("Running model {}".format(f))
            #print("Args: {}".format(" ".join([str(i) for i in args])))
            main(args)
            successful.append(f)

        except KeyboardInterrupt:
            print("Keyboard interrupt encountered. Stopping.")
            break
        except Exception as e:
            print("Encountered exception while running model {}: {}".format(f, e.args[0]))

    print("All models complete")
    print("Successfully completed: {}".format(", ".join(successful)))

