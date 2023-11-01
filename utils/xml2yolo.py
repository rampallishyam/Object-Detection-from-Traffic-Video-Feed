import xml.etree.ElementTree as ET
__all__ = ['xml2yolo']

vehicle_classes = ['motorcycle','autorickshaw','car','bus','truck']

def xml2yolo(input_xml, output_txt):

# Parse the XML file
    tree = ET.parse(input_xml)
    root = tree.getroot()

    # Open the TXT file for writing
    with open(output_txt, 'w') as txt_file:
        for img_size in root.iter('size'):
            size_child = [child for child in img_size]
            width = int(size_child[0].text)
            height = int(size_child[1].text)

        for object in root.iter('object'):
            obj_child = [child for child in object]
            if obj_child[0].text in vehicle_classes:
                ind = vehicle_classes.index(obj_child[0].text)
                box_dim = [child for child in obj_child[1]]
                cntr_x = (float(box_dim[0].text) + float(box_dim[2].text)) / (width*2.0)
                cntr_y = (float(box_dim[1].text) + float(box_dim[3].text)) / (height*2.0)
                w = (float(box_dim[2].text) - float(box_dim[0].text))                
                h = (float(box_dim[1].text) - float(box_dim[3].text)) 

                if (w*h>=1500):
                    w = float(w/width)
                    h = float(h/height)
                    txt_file.write(str(ind)+' ' + str(cntr_x) + " " + str(cntr_y) +  " " + str(w) + " " + str(h) + '\n')
