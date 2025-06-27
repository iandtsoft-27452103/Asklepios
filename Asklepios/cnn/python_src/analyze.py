import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import policy
import value
import pv_nn
import file
import rank
import piece
import bitop
import makemove
import feature
import gencap
import gennocap
import gendrop
import genevasion
import attack
import sfen
import csa
import move
import color as C
import numpy

class analyze:
    def __init__(self):
        self.date_of_game = '2020/11/22'
        self.name_of_category = '第70回NHK杯'
        self.black_player = '木村一基九段'
        self.white_player = '藤井聡太二冠'
        self.engine_name = 'Gift Ver.1.0.0'

    #CSA形式のファイルをこのソフトが読み込める形式に変換する
    def csa_to_record(self):
        in_file_name = '20230709▲藤森哲也五段対△山崎隆之八段.csa'
        out_file_name = '20230709_nhk_hai.txt'

        f_in = open(in_file_name, 'r', 1, 'SHIFT_JIS')
        f_out = open(out_file_name, 'w', 1, 'UTF-8')
        li = []
        color = 0
        ply = 0
        for line in f_in:
            if line[0] == '+' or line[0] == '-':
                if len(line) == 2:
                    continue
                str_csa_move = line
                line = line.replace('\n', '')
                line = line.replace('+', '')
                line = line.replace('-', '')
                li.append(line)
                color ^= 1
                ply += 1

            if line == '%TORYO\n':
                winner = color ^ 1

        out = ''

        if winner == 0:
            out += 'B,'
        elif winner == 1:
            out += 'W,'
        else:
            out += 'D,'

        out += str(ply)

        for line in li:
            out += ','
            out += line

        out += '\n'

        f_out.write(out)

        f_in.close()
        f_out.close()

    #Value Networkを使って分析する
    def analyze_value(self, record, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        at = ft.at
        #c = C.color()
        #cls_cap = gencap.gencap(bo, bi, pc, at, c)
        #cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        #cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        #cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        f = open('analize_result_value.txt', 'w', 1, 'UTF-8')

        self.out_header(f)

        device = torch.device("cuda:0")
        model = value.value()
        model.load_state_dict(torch.load('model_value.pth'))
        model = model.to(device)

        color = 0
        bo.init_board(bo.board_default, bo.hand_default, bi, 0)
        for ply in range(1, len(record.moves) + 1):
            current = record.moves[ply - 1]
            ma.makemove(bo, current, ply, bi, pc, color)
            color = color ^ 1
            li = []
            fe = ft.make_input_features1(bo)
            li.append(fe)
            li2 = []
            fe = ft.make_input_features2(bo)
            li2.append(fe)
            li3 = []
            fe = ft.make_input_features3(bo, color)
            li3.append(fe)

            li = numpy.array(li)
            li2 = numpy.array(li2)
            li3 = numpy.array(li3)

            model.eval()

            with torch.no_grad():
                x1 = torch.tensor(li, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(li2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(li3, dtype = torch.float)
                x3 = x3.to(device)
                model.batch_size = 1
                y = model.forward(x1, x2, x3)
                y = y.to("cpu")

                correct_move = current
                correct_label, direc = ft.make_output_labels(bo, correct_move)
                z = y.sigmoid()
                s = "-"
                if color == 1:
                    z = 1 - z
                    s = "+"
                z = z * 100
                z = round(float(z))
                f.write("ply=" + str(ply) + "   " + s + record.str_moves[ply - 1] + "   " + str(z) + "%, " + str(100 - z) + "%\n")
                #print("ply=" + str(ply) + "   " + s + record.str_moves[ply - 1] + "   " + str(z) + "%, " + str(100 - z) + "%")
        f.write('\n')
        self.out_footer(f)
        f.close()

    def analyze_policy(self, record, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)
        cls_csa = csa.csa()

        f = open('analize_result_policy.txt', 'w', 1, 'UTF-8')

        self.out_header(f)

        device = torch.device("cuda:0")
        model = policy.policy()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)

        total_move_count = 0
        total_match_count = 0

        color = 0
        move_count = 0
        match_count = 0
        black_move_count = 0
        black_match_count = 0
        white_move_count = 0
        white_match_count = 0
        bo.init_board(bo.board_default, bo.hand_default, bi, 0)
        for ply in range(1, len(record.moves) + 1):
            current = record.moves[ply - 1]
            li = []
            fe = ft.make_input_features1(bo)
            li.append(fe)
            li2 = []
            fe = ft.make_input_features2(bo)
            li2.append(fe)
            li3 = []
            fe = ft.make_input_features3(bo, color)
            li3.append(fe)

            li = numpy.array(li)
            li2 = numpy.array(li2)
            li3 = numpy.array(li3)

            model.eval()

            with torch.no_grad():
                x1 = torch.tensor(li, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(li2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(li3, dtype = torch.float)
                x3 = x3.to(device)
                y = model.forward(x1, x2, x3)
                y = y.to("cpu")

                moves = []
                if color == 0:
                    if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                        cls_cap.b_gen_captures(moves)
                        cls_nocap.b_gen_nocaptures(moves)
                        cls_drop.b_gen_drop(moves)
                    else:
                        cls_eva.b_gen_evasion(moves)
                else:
                    if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                        cls_cap.w_gen_captures(moves)
                        cls_nocap.w_gen_nocaptures(moves)
                        cls_drop.w_gen_drop(moves)
                    else:
                        cls_eva.w_gen_evasion(moves)

                correct_move = current
                correct_label, direc = ft.make_output_labels(bo, correct_move)
                z = y.data[0].tolist()

                correct_digit = z[direc][bo.rank_table[correct_move.ito]][bo.file_table[correct_move.ito]]

                digits = []
                flag = False
                for i in range(len(moves)):
                    a, b = ft.make_output_labels(bo, moves[i])
                    c = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                    digits.append(c)
                    if c > correct_digit:
                        com_move = moves[i]
                        flag = True

                ma.makemove(bo, correct_move, ply, bi, pc, color)

                if flag == False:
                    com_move = correct_move
                    match_count += 1
                    if color == 0:
                        black_match_count += 1
                    else:
                        white_match_count += 1
            if color == 0:
                s = "+"
                black_move_count += 1
            else:
                s = "-"
                white_move_count += 1
            if flag == False:
                s2 = "○"
            else:
                s2 = "×"
            str_com_move = s + cls_csa.board_to_csa(bo, pc, com_move)
            print("ply=" + str(ply) + "   pro= " + s + record.str_moves[ply - 1] + ",   com= " + str_com_move + ",   result= " + s2)
            f.write("ply=" + str(ply) + "   pro= " + s + record.str_moves[ply - 1] + ",   com= " + str_com_move + ",   result= " + s2 + "\n")
            color = color ^ 1
            move_count += 1

        black_matching_rate = black_match_count / black_move_count
        black_matching_rate *= 100
        black_matching_rate = '{a:.2f}'.format(a=black_matching_rate)
        white_matching_rate = white_match_count / white_move_count
        white_matching_rate *= 100
        white_matching_rate = '{a:.2f}'.format(a=white_matching_rate)
        matching_rate = match_count / move_count
        matching_rate *= 100
        matching_rate = '{a:.2f}'.format(a=matching_rate)
        print("")
        f.write("\n")
        print(match_count, "/", move_count, " ", matching_rate + "%")
        f.write('先手一致率：' + str(black_match_count) + " / " + str(black_move_count) + " " + black_matching_rate + "%\n")
        f.write('\n')
        f.write('後手一致率：' + str(white_match_count) + " / " + str(white_move_count) + " " + white_matching_rate + "%\n")
        f.write('\n')
        f.write('全体一致率：' + str(match_count) + " / " + str(move_count) + " " + matching_rate + "%\n")
        f.write('\n')
        self.out_footer(f)
        print("")
        #f.write("\n")

        f.close()

    def analyze_pv_nn(self, record, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)
        cls_csa = csa.csa()

        f = open('analize_result_pv_nn.txt', 'w', 1, 'UTF-8')

        self.out_header(f)

        device = torch.device("cuda:0")
        model = pv_nn.pv_nn()
        model.load_state_dict(torch.load('model_pv_nn.pth'))
        model = model.to(device)

        total_move_count = 0
        total_match_count = 0

        color = 0
        move_count = 0
        match_count = 0
        black_move_count = 0
        black_match_count = 0
        white_move_count = 0
        white_match_count = 0
        bo.init_board(bo.board_default, bo.hand_default, bi, 0)
        for ply in range(1, len(record.moves) + 1):
            current = record.moves[ply - 1]
            li = []
            fe = ft.make_input_features1(bo)
            li.append(fe)
            li2 = []
            fe = ft.make_input_features2(bo)
            li2.append(fe)
            li3 = []
            fe = ft.make_input_features3(bo, color)
            li3.append(fe)

            li = numpy.array(li)
            li2 = numpy.array(li2)
            li3 = numpy.array(li3)

            model.eval()

            with torch.no_grad():
                x1 = torch.tensor(li, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(li2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(li3, dtype = torch.float)
                x3 = x3.to(device)
                model.batch_size = 1
                y, _ = model.forward(x1, x2, x3)
                y = y.to("cpu")

                moves = []
                if color == 0:
                    if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                        cls_cap.b_gen_captures(moves)
                        cls_nocap.b_gen_nocaptures(moves)
                        cls_drop.b_gen_drop(moves)
                    else:
                        cls_eva.b_gen_evasion(moves)
                else:
                    if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                        cls_cap.w_gen_captures(moves)
                        cls_nocap.w_gen_nocaptures(moves)
                        cls_drop.w_gen_drop(moves)
                    else:
                        cls_eva.w_gen_evasion(moves)

                correct_move = current
                correct_label, direc = ft.make_output_labels(bo, correct_move)
                z = y.data[0].tolist()

                correct_digit = z[direc][bo.rank_table[correct_move.ito]][bo.file_table[correct_move.ito]]

                digits = []
                flag = False
                for i in range(len(moves)):
                    a, b = ft.make_output_labels(bo, moves[i])
                    c = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                    digits.append(c)
                    if c > correct_digit:
                        com_move = moves[i]
                        flag = True

                ma.makemove(bo, correct_move, ply, bi, pc, color)

                if flag == False:
                    com_move = correct_move
                    match_count += 1
                    if color == 0:
                        black_match_count += 1
                    else:
                        white_match_count += 1
            if color == 0:
                s = "+"
                black_move_count += 1
            else:
                s = "-"
                white_move_count += 1
            if flag == False:
                s2 = "○"
            else:
                s2 = "×"
            str_com_move = s + cls_csa.board_to_csa(bo, pc, com_move)
            print("ply=" + str(ply) + "   pro= " + s + record.str_moves[ply - 1] + ",   com= " + str_com_move + ",   result= " + s2)
            f.write("ply=" + str(ply) + "   pro= " + s + record.str_moves[ply - 1] + ",   com= " + str_com_move + ",   result= " + s2 + "\n")
            color = color ^ 1
            move_count += 1

        black_matching_rate = black_match_count / black_move_count
        black_matching_rate *= 100
        black_matching_rate = '{a:.2f}'.format(a=black_matching_rate)
        white_matching_rate = white_match_count / white_move_count
        white_matching_rate *= 100
        white_matching_rate = '{a:.2f}'.format(a=white_matching_rate)
        matching_rate = match_count / move_count
        matching_rate *= 100
        matching_rate = '{a:.2f}'.format(a=matching_rate)
        print("")
        f.write("\n")
        print(match_count, "/", move_count, " ", matching_rate + "%")
        f.write('先手一致率：' + str(black_match_count) + " / " + str(black_move_count) + " " + black_matching_rate + "%\n")
        f.write('\n')
        f.write('後手一致率：' + str(white_match_count) + " / " + str(white_move_count) + " " + white_matching_rate + "%\n")
        f.write('\n')
        f.write('全体一致率：' + str(match_count) + " / " + str(move_count) + " " + matching_rate + "%\n")
        f.write('\n')
        self.out_footer(f)
        print("")
        #f.write("\n")

        f.close()

    def out_header(self, f):
        f.write('対局日：' + self.date_of_game + '\n')
        f.write('\n')
        f.write('棋戦名：' + self.name_of_category + '\n')
        f.write('\n')
        f.write('先手：' + self.black_player + '\n')
        f.write('\n')
        f.write('後手：' + self.white_player + '\n')
        f.write('\n')

    def out_footer(self, f):
        f.write('解析エンジン名：' + self.engine_name)
        f.write('\n')
