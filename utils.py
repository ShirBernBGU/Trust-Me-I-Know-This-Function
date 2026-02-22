import json
import re

import clang
import clang.cindex as cindex

cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-20.so.1")  # Set the path to libclang.so

def load_jsonl_dataset(path : str) -> str:
    data = []

    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def extract_function_name(code):                                                                                                         # Initialize the Clang compiler index
    index = cindex.Index.create()                                                                                                    
    # Parse the code as a translation unit
    translation_unit = index.parse("temp.c", unsaved_files=[("temp.c", code)])                        
    current_file_path = translation_unit.cursor.spelling
    # Find the first function declaration in the translation unit
    for node in translation_unit.cursor.walk_preorder():
        if node.kind == cindex.CursorKind.FUNCTION_DECL and node.location.file and node.location.file.name == current_file_path:
            return node.spelling                                                                                                     
    return None

def extract_function2(out_file : str, function_name : str, second_function_name : str = "", is_merged : bool = False, opaque : bool = False, encode_branches : bool = False, extract_helpers : bool = True, extract_only_helpers : bool = False) -> None:
    target_file = out_file

    try:
        with open(target_file, "r") as f:
            lines = f.readlines()
    except:
        print("Error opening the file. Skipping to the next file")
        return str()

    real_function = str()
    tigress_data = str()
    extracted_function = str()
    temp = str()

    append_to_beginning = False
    do_extract = False
    do_extract_tigress_data = False
    do_extract_real_function = False
    extracted___2_bf_1 = False
    __2_bf_1_ids = []

    # read line by line until the function signature comment is found
    for i in range(len(lines)):
        if (
            "/* BEGIN" in lines[i-1] and ("LOC=UNKNOWN " in lines[i-1] or "LOC=generated_primitives/" + function_name + ".c" in lines[i-1]) and
            not re.search(r"( __bswap_(16|32|64) | __uint(16|32|64)_identity)", lines[i-1])
            ):

            if not re.search(r"/\* BEGIN FUNCTION-(DECL|DEF) " + function_name, lines[i-1]):
                do_extract_tigress_data = True

            do_extract = True

        if "/* END" in lines[i] and ("LOC=UNKNOWN " in lines[i] or "LOC=generated_primitives/" + function_name + ".c" in lines[i]):
            do_extract_tigress_data = False
            do_extract = False
            
        if "/* BEGIN TYPEDEF size_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True
            
        if "/* END TYPEDEF size_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False
        
        if "/* BEGIN TYPEDEF FILE" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True
            
        if "/* END TYPEDEF FILE" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN STRUCT timespec" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END STRUCT timespec" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN TYPEDEF pthread_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END TYPEDEF pthread_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False
        
        if "/* BEGIN TYPEDEF __time_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END TYPEDEF __time_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN TYPEDEF time_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END TYPEDEF time_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN TYPEDEF __id_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END TYPEDEF __id_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN " in lines[i-1] and ("min_cost_to_hire_workers.c" in lines[i-1] or "generated_primitives_reasoning/visvalingam_whyatt.c" in lines[i-1] or "generated_primitives_reasoning/base64url_decode.c" in lines[i-1] or "generated_primitives_reasoning/range_gcd_sqrt.c" in lines[i-1] or "generated_primitives_reasoning/shortest_cycle_length.c" in lines[i-1] or "generated_primitives_reasoning/sliding_window_min.c" in lines[i-1]):
            do_extract_tigress_data = True
            do_extract = True

        if "/* END " in lines[i] and ("min_cost_to_hire_workers.c" in lines[i] or "generated_primitives_reasoning/visvalingam_whyatt.c" in lines[i] or "generated_primitives_reasoning/base64url_decode.c" in lines[i] or "generated_primitives_reasoning/range_gcd_sqrt.c" in lines[i] or "generated_primitives_reasoning/shortest_cycle_length.c" in lines[i] or "generated_primitives_reasoning/sliding_window_min.c" in lines[i]):
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN " in lines[i-1] and "generated_primitives_reasoning" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END " in lines[i] and "generated_primitives_reasoning" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if "/* BEGIN TYPEDEF __syscall_slong_t" in lines[i-1]:
            do_extract_tigress_data = True
            do_extract = True

        if "/* END TYPEDEF __syscall_slong_t" in lines[i]:
            do_extract_tigress_data = False
            do_extract = False

        if not "#line" in lines[i]:
            if append_to_beginning == True:
                temp += lines[i]

            if do_extract_tigress_data == True:
                tigress_data += lines[i]

            if do_extract_real_function == True:
                real_function += lines[i]

            if do_extract == True:
                extracted_function += lines[i]

    obfuscated_function_name = extract_function_name(real_function)

    if extract_only_helpers:
        return tigress_data

    if not extract_helpers:
       return (obfuscated_function_name, real_function)

    return (obfuscated_function_name, "\n".join(extracted_function.split("\n"))) # remove the first comment and the additional line
    