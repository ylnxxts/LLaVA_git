import json

if __name__ == '__main__':
    json_root=r'/data/majunpeng/LLaVA/playground/data/llava_v1_5_mix665k.json'
    with open(json_root, 'r') as f:
        json_info = json.load(f)
    print(json_info[0])
    save_json = []
    for content in json_info:
        try:
            print(content["image"])
            if content["image"][0:7] == "textvqa":
                save_json.append(content)
        except:
            continue
            
    print(len(save_json))
    # save_info=json_info[:1000]+json_info[480000:481000]+json_info[620000:621000]
    # with open('textvqa.json', 'w') as fp:
    #     json.dump(save_json, fp, indent=4)

