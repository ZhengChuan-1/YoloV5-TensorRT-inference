#include <iostream>
#include <vector>
#include <list>
using namespace std;

string names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "'skis'", "'snowboard'", "'sports ball'", "'kite'", "'baseball bat'", "'baseball glove'", "'skateboard'", "'surfboard'",
        "'tennis racket'", "'bottle'", "'wine glass'", "'cup'", "'fork'", "'knife'", "'spoon'", "'bowl'", "'banana'", "'apple'",
        "'sandwich'", "'orange'", "'broccoli'", "'carrot'", "'hot dog'", "'pizza'", "'donut'", "'cake'", "'chair'", "'couch'",
        "'potted plant'", "'bed'", "'dining table'", "'toilet'", "'tv'", "'laptop'", "'mouse'", "'remote'", "'keyboard'", "'cell phone'",
        "'microwave'", "'oven'", "'toaster'", "'sink'", "'refrigerator'", "'book'", "'clock'", "'vase'", "'scissors'", "'teddy bear'",
        "'hair drier'", "'toothbrush'"};

struct BOX
{
    float x;
    float y;
    float width;
    float height;
};

struct Object
{
    BOX box;
    int label;
    float confidence;
};

bool cmp(Object &obj1, Object &obj2){
    return obj1.confidence > obj2.confidence;
}

vector<list<Object>> NMS(std::vector<Object> objs, float iou_thres = 0.45){
    //第一步：将所有矩形框按照不同的类别标签分组，组内按照置信度高低得分进行排序；
    
    list<Object> obj_l;
    vector<list<Object>> NMS_List;
    int a = 0;
    for(int i = 0; i < 80; i++){
        for(auto j : objs)
        {
            if(j.label == i){
                obj_l.push_back(j);
                obj_l.sort(cmp);        //依据置信度升序排序
                a = 1;
            }
        }
        if(a == 1){
            NMS_List.push_back(std::move(obj_l));
            a = 0;
            }
    }

    //第二步：计算IOU
    float x1, y1, x1_w, y1_h,x2, y2, x2_w, y2_h;
    float x_box, y_box, x_w_box, y_h_box, w_box, h_box;
    float S1,S2,SBOX,res_iou;

    int row = NMS_List.size();  //行数     列数：NMS_List[0].size()
    int tmp;
    for(int i = 0; i < row ; i++)  //不同分类的循环
    {
        tmp = 0;
        list<Object>::iterator it = NMS_List[i].begin();
        while(it != --NMS_List[i].end()){
            x1 = it->box.x;
            y1 = it->box.y;
            x1_w = x1 + it->box.width;
            y1_h = y1 + it->box.height;
            while(it != --NMS_List[i].end())
            {
                it++;
                x2 = it->box.x;
                y2 = it->box.y;
                x2_w = x2 + it->box.width;
                y2_h = y2 + it->box.height;
                //交集左上角坐标x_box,y_box  框1-x1和框2-x2的最大值   框1-y1和框2-y2的最大值
                x_box = std::max(x1, x2);
                y_box = std::max(y1, y2);
                //交集右下角坐标x_w_box,y_h_box  框1-x1_w和框2-x2_w的最小值  框1-y1_h和框2-y2_h的最小值
                x_w_box = std::min(x1_w, x2_w);
                y_h_box = std::min(y1_h, y2_h);
                //交集框宽高
                w_box = x_w_box - x_box;
                h_box = y_h_box - y_box;
                //无交集情况
                if(w_box <= 0 || h_box <= 0)
                {
                    it = NMS_List[i].erase(it);
                    if(it == NMS_List[i].end()){break;}
                    it--;
                    continue;
                }
                //有交集，计算IOU
                S1 = (x1_w - x1) * (y1_h - y1);
                S2 = (x2_w - x2) * (y2_h - y2);
                SBOX = w_box * h_box;
                if((res_iou = SBOX / (S1 + S2 - SBOX)) > iou_thres){
                    it = NMS_List[i].erase(it);
                    if(it == NMS_List[i].end()){break;}
                    it--;
                }

            }
            it = NMS_List[i].begin();
            if(it == --NMS_List[i].end()){break;}
            tmp++;
            for(int z = 0; z < tmp; z++){
                it++;
                if(it == --NMS_List[i].end()){break;}
            }
        }
    }

    return NMS_List;
}