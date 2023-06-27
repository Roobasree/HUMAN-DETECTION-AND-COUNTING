import cv2
from persondetection import DetectorAPI
import argparse


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())
    return args
args = argsParser()
writer = None
if args['output'] is not None:
    writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

max_count2 = 0
framex2 = []
county2 = []
max2 = []
avg_acc2_list = []
max_avg_acc2_list = []
max_acc2 = 0
max_avg_acc2 = 0
#video input
video = cv2.VideoCapture(0)
odapi = DetectorAPI()
threshold = 0.7

check, frame = video.read()
if check == False:
    print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')

x2 = 0
while video.isOpened():
    # check is True if reading was successful
    check, frame = video.read()
    if (check == True):
        img = cv2.resize(frame, (800, 500))
        boxes, scores, classes, num = odapi.processFrame(img)
        person = 0
        acc = 0
        for i in range(len(boxes)):
            # print(boxes)
            # print(scores)
            # print(classes)
            # print(num)
            # print()
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 255), 1)  # (75,0,130),
                acc += scores[i]
                if (scores[i] > max_acc2):
                    max_acc2 = scores[i]

        if (person > max_count2 and max_count2!=0):
            max_count2 = person
        county2.append(person)
        x2 += 1
        framex2.append(x2)
        if (person >= 1):
            avg_acc2_list.append(acc / person)
            if ((acc / person) > max_avg_acc2):
                max_avg_acc2 = (acc / person)
        else:
            avg_acc2_list.append(acc)

        if writer is not None:
            writer.write(img)
        lpc_count = person
        opc_count = max_count2
        lpc_txt = "Live Person Count: {}".format(lpc_count)
        opc_txt = "Overall Person Count:{}".format(opc_count)
        cv2.putText(img, lpc_txt, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(img, opc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Human Detection from Video", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

for i in range(len(framex2)):
    max2.append(max_count2)
    max_avg_acc2_list.append(max_avg_acc2)