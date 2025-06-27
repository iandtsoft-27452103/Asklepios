import numpy as np
#import inout
#import result
import bitop
import label
import direction
import piece
import attack

class feature:
    def __init__(self, bo):
        self.bi = bitop.bitop(bo)
        self.bb_full = self.bi.bb_ini()
        bb = self.bi.bb_ini()
        for i in range(len(bo.board)):
            bb = self.bi.xor(i, bb)
        self.bb_full = bb
        self.l = label.label()
        self.pc = piece.piece()
        self.di = direction.direction()
        self.at = attack.attack()
        self.label_list = [[[0 for k in range(bo.square_nb)] for j in range(bo.square_nb + 7)] for k in range(2)]
        self.label_list_init(bo)

    def label_list_init(self, bo):
        magic_number = 1
        for flag_promo in range(2):
            for ifrom in range(bo.square_nb + 7): #7は駒打ちの種別（歩・香・桂・銀・金・角・飛）
                for ito in range(bo.square_nb):
                    if ifrom > bo.square_nb - 1:
                        self.label_list[flag_promo][ifrom][ito] = magic_number
                        magic_number += 1
                    else:
                        if self.at.adirec[ifrom][ito] != self.di.direc_misc:
                            self.label_list[flag_promo][ifrom][ito] = magic_number
                            magic_number += 1
                        else:
                            d = ito - ifrom
                            if d == self.di.knight_north_west or d == self.di.knight_north_east or d == self.di.knight_south_east or d == self.di.knight_south_west:
                                self.label_list[flag_promo][ifrom][ito] = magic_number
                                magic_number += 1
        #print(str(magic_number))

    def make_input_features1(self, bo):
        features = []
        #先後の駒
        for i in range(2):
            #盤上の駒
            bb = bo.bb_pawn[i]
            self.loop(bo, features, bb)
            bb = bo.bb_lance[i]
            self.loop(bo, features, bb)
            bb = bo.bb_knight[i]
            self.loop(bo, features, bb)
            bb = bo.bb_silver[i]
            self.loop(bo, features, bb)
            bb = bo.bb_gold[i]
            self.loop(bo, features, bb)
            bb = bo.bb_bishop[i]
            self.loop(bo, features, bb)
            bb = bo.bb_rook[i]
            self.loop(bo, features, bb)
            ft = np.zeros(bo.nfile * bo.nrank)
            ft[bo.sq_king[i]] = 1
            features.append(ft.reshape((bo.nfile, bo.nrank)))
            bb = bo.bb_pro_pawn[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_lance[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_knight[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_silver[i]
            self.loop(bo, features, bb)
            bb = bo.bb_horse[i]
            self.loop(bo, features, bb)
            bb = bo.bb_dragon[i]
            self.loop(bo, features, bb)

            #自玉の近傍8マスにある金銀
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i]]
            bb = self.bi.bb_or(bo.bb_silver[i], bo.bb_gold[i])
            bb = self.bi.bb_and(bb, bb_king_attacks)
            self.loop(bo, features, bb)

        return features

    def make_input_features2(self, bo):
        features = []
        for i in range(2):
            #持ち駒
            self.loop2(bo, features, bo.hand_pawn[i], 6)#6枚まで評価する
            self.loop2(bo, features, bo.hand_lance[i], bo.nlance)
            self.loop2(bo, features, bo.hand_knight[i], bo.nknight)
            self.loop2(bo, features, bo.hand_silver[i], bo.nsilver)
            self.loop2(bo, features, bo.hand_gold[i], bo.ngold)
            self.loop2(bo, features, bo.hand_bishop[i], bo.nbishop)
            self.loop2(bo, features, bo.hand_rook[i], bo.nrook)

        return features

    def make_input_features3(self, bo, color):
        features = []
        for i in range(2):

            #駒台に金銀が3枚以上あるか？
            cnt = bo.hand_silver[i] + bo.hand_gold[i]
            if cnt >= 3:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #駒台に桂が2枚以上あるか？
            if bo.hand_knight[i] >= 2:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #駒台に香が2枚以上あるか？
            if bo.hand_lance[i] >= 2:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #歩切れか？
            if bo.hand_pawn[i] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #自分だけ龍を作っているか？
            if bo.bb_dragon[i] != 0 and bo.bb_dragon[i ^ 1] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #自分だけ馬を作っているか？
            if bo.bb_horse[i] != 0 and bo.bb_horse[i ^ 1] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #自玉に王手がかかっているか？
            if self.at.is_attacked(bo, bo.sq_king[i], i ^ 1) != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #金銀を6枚以上持っているか？
            cnt = bo.hand_silver[i] + bo.hand_gold[i]
            cnt += self.bi.popu_count(bo.bb_silver[i])
            cnt += self.bi.popu_count(bo.bb_gold[i])
            cnt += self.bi.popu_count(bo.bb_pro_silver[i])
            if cnt >= 6:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #龍が敵陣にあるか
            li = []
            bb = bo.bb_dragon[i]
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                li.append(sq)

            flag = False
            for j in range(len(li)):
                if i == 0:
                    if li[j] < 27:
                        flag = True
                        break
                else:
                    if li[j] > 53:
                        flag = True
                        break
            if flag == True:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #自玉の8近傍に自分の馬・角が長い利きをつけているか
            li = []
            bb = bo.bb_bh[i]
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                li.append(sq)

            bb = self.bi.bb_ini()
            for j in range(len(li)):
                bb_attacks = self.at.get_bishop_attacks(bo.bb_occupied, li[j])
                bb = self.bi.bb_or(bb, bb_attacks)
            
            bb2 = self.bi.bb_ini()
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i]]
            bb2 = self.bi.bb_and(bb, bb_king_attacks)

            if bb2 != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

            #敵玉の8近傍に自分の馬・角が長い利きをつけているか
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i ^ 1]]
            bb2 = self.bi.bb_ini()
            bb2 = self.bi.bb_and(bb, bb_king_attacks)

            if bb2 != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

        #手番
        if color == 0:
            ft = np.ones(bo.nfile * bo.nrank)
        else:
            ft = np.zeros(bo.nfile * bo.nrank)
        features.append(ft.reshape((bo.nfile, bo.nrank)))

        return features

    def make_input_features4(self, bo, color):
        features = []
        #先後の駒
        for i in range(2):
            #盤上の駒
            bb = bo.bb_pawn[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_lance[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_knight[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_silver[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_gold[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_bishop[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_rook[i]
            self.loop3(bo, features, bb)
            ft = np.zeros(bo.nfile * bo.nrank)
            ft[bo.sq_king[i]] = 1
            features.append(ft)
            bb = bo.bb_pro_pawn[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_pro_lance[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_pro_knight[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_pro_silver[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_horse[i]
            self.loop3(bo, features, bb)
            bb = bo.bb_dragon[i]
            self.loop3(bo, features, bb)

            #自玉の近傍8マスにある金銀
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i]]
            bb = self.bi.bb_or(bo.bb_silver[i], bo.bb_gold[i])
            bb = self.bi.bb_and(bb, bb_king_attacks)
            self.loop3(bo, features, bb)

            #持ち駒
            self.loop4(bo, features, bo.hand_pawn[i], 6)#6枚まで評価する
            self.loop4(bo, features, bo.hand_lance[i], bo.nlance)
            self.loop4(bo, features, bo.hand_knight[i], bo.nknight)
            self.loop4(bo, features, bo.hand_silver[i], bo.nsilver)
            self.loop4(bo, features, bo.hand_gold[i], bo.ngold)
            self.loop4(bo, features, bo.hand_bishop[i], bo.nbishop)
            self.loop4(bo, features, bo.hand_rook[i], bo.nrook)

            #駒台に金銀が3枚以上あるか？
            cnt = bo.hand_silver[i] + bo.hand_gold[i]
            if cnt >= 3:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #駒台に桂が2枚以上あるか？
            if bo.hand_knight[i] >= 2:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #駒台に香が2枚以上あるか？
            if bo.hand_lance[i] >= 2:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #歩切れか？
            if bo.hand_pawn[i] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #自分だけ龍を作っているか？
            if bo.bb_dragon[i] != 0 and bo.bb_dragon[i ^ 1] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #自分だけ馬を作っているか？
            if bo.bb_horse[i] != 0 and bo.bb_horse[i ^ 1] == 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #自玉に王手がかかっているか？
            if self.at.is_attacked(bo, bo.sq_king[i], i ^ 1) != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #金銀を6枚以上持っているか？
            cnt = bo.hand_silver[i] + bo.hand_gold[i]
            cnt += self.bi.popu_count(bo.bb_silver[i])
            cnt += self.bi.popu_count(bo.bb_gold[i])
            cnt += self.bi.popu_count(bo.bb_pro_silver[i])
            if cnt >= 6:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #龍が敵陣にあるか
            li = []
            bb = bo.bb_dragon[i]
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                li.append(sq)

            flag = False
            for j in range(len(li)):
                if i == 0:
                    if li[j] < 27:
                        flag = True
                        break
                else:
                    if li[j] > 53:
                        flag = True
                        break
            if flag == True:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #自玉の8近傍に自分の馬・角が長い利きをつけているか
            li = []
            bb = bo.bb_bh[i]
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                li.append(sq)

            bb = self.bi.bb_ini()
            for j in range(len(li)):
                bb_attacks = self.at.get_bishop_attacks(bo.bb_occupied, li[j])
                bb = self.bi.bb_or(bb, bb_attacks)
            
            bb2 = self.bi.bb_ini()
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i]]
            bb2 = self.bi.bb_and(bb, bb_king_attacks)

            if bb2 != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

            #敵玉の8近傍に自分の馬・角が長い利きをつけているか
            bb_king_attacks = self.at.abb_king_attacks[bo.sq_king[i ^ 1]]
            bb2 = self.bi.bb_ini()
            bb2 = self.bi.bb_and(bb, bb_king_attacks)

            if bb2 != 0:
                ft = np.ones(bo.nfile * bo.nrank)
            else:
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

        #手番
        if color == 0:
            ft = np.ones(bo.nfile * bo.nrank)
        else:
            ft = np.zeros(bo.nfile * bo.nrank)
        features.append(ft)

        return features

    def loop(self, bo, features, bb):
        ft = np.zeros(bo.nfile * bo.nrank)
        while bb != 0:
            sq = self.bi.first_one(bb)
            bb = self.bi.xor(sq, bb)
            ft[sq] = 1
        features.append(ft.reshape((bo.nfile, bo.nrank)))

    def loop2(self, bo, features, hand, count):
        h = hand
        for i in range(count):
            if h > 0:
                #持ち駒があったら1埋め
                ft = np.ones(bo.nfile * bo.nrank)
                h -= 1
            else:
                #持ち駒がなかったら0埋め
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

    def loop3(self, bo, features, bb):
        ft = np.zeros(bo.nfile * bo.nrank)
        while bb != 0:
            sq = self.bi.first_one(bb)
            bb = self.bi.xor(sq, bb)
            ft[sq] = 1
        features.append(ft)

    def loop4(self, bo, features, hand, count):
        h = hand
        for i in range(count):
            if h > 0:
                #持ち駒があったら1埋め
                ft = np.ones(bo.nfile * bo.nrank)
                h -= 1
            else:
                #持ち駒がなかったら0埋め
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft)

    def make_output_labels(self, bo, m):
        if m.ifrom < bo.square_nb:
            #通常の駒の移動→ifromには80以内の数が入っている
            d = m.ito - m.ifrom
            if m.flag_promo == 0:
                if m.piece_to_move != self.pc.knight:
                    #桂馬以外を動かす手の場合
                    if d == self.di.north_west or d == (self.di.north_west * 2) or d == (self.di.north_west * 3) or d == (self.di.north_west * 4) or d == (self.di.north_west * 5) or d == (self.di.north_west * 6) or d == (self.di.north_west * 7) or d == (self.di.north_west * 8):
                        move_direction = self.l.UP_LEFT
                    elif d == self.di.north or d == (self.di.north * 2) or d == (self.di.north * 3) or d == (self.di.north * 4) or d == (self.di.north * 5) or d == (self.di.north * 6) or d == (self.di.north * 7) or d == (self.di.north * 8):
                        move_direction = self.l.UP
                    elif d == self.di.north_east or d == (self.di.north_east * 2) or d == (self.di.north_east * 3) or d == (self.di.north_east * 4) or d == (self.di.north_east * 5) or d == (self.di.north_east * 6) or d == (self.di.north_east * 7) or d == (self.di.north_east * 8):
                        move_direction = self.l.UP_RIGHT
                    elif d == self.di.east or d == (self.di.east * 2) or d == (self.di.east * 3) or d == (self.di.east * 4) or d == (self.di.east * 5) or d == (self.di.east * 6) or d == (self.di.east * 7) or d == (self.di.east * 8):
                        move_direction = self.l.RIGHT
                    elif d == self.di.south_east or d == (self.di.south_east * 2) or d == (self.di.south_east * 3) or d == (self.di.south_east * 4) or d == (self.di.south_east * 5) or d == (self.di.south_east * 6) or d == (self.di.south_east * 7) or d == (self.di.south_east * 8):
                        move_direction = self.l.DOWN_RIGHT
                    elif d == self.di.south or d == (self.di.south * 2) or d == (self.di.south * 3) or d == (self.di.south * 4) or d == (self.di.south * 5) or d == (self.di.south * 6) or d == (self.di.south * 7) or d == (self.di.south * 8):
                        move_direction = self.l.DOWN
                    elif d == self.di.south_west or d == (self.di.south_west * 2) or d == (self.di.south_west * 3) or d == (self.di.south_west * 4) or d == (self.di.south_west * 5) or d == (self.di.south_west * 6) or d == (self.di.south_west * 7) or d == (self.di.south_west * 8):
                        move_direction = self.l.DOWN_LEFT
                    elif d == self.di.west or d == (self.di.west * 2) or d == (self.di.west * 3) or d == (self.di.west * 4) or d == (self.di.west * 5) or d == (self.di.west * 6) or d == (self.di.west * 7) or d == (self.di.west * 8):
                        move_direction = self.l.LEFT
                else:
                    #桂馬を動かす手の場合
                    if d == self.di.knight_north_west:
                        move_direction = self.l.UP_LEFT_KNIGHT
                    elif d == self.di.knight_north_east:
                        move_direction = self.l.UP_RIGHT_KNIGHT
                    elif d == self.di.knight_south_east:
                        move_direction = self.l.DOWN_RIGHT_KNIGHT
                    elif d == self.di.knight_south_west:
                        move_direction = self.l.DOWN_LEFT_KNIGHT
            else:
                if m.piece_to_move != self.pc.knight:
                    #桂馬以外を動かす手の場合
                    if d == self.di.north_west or d == (self.di.north_west * 2) or d == (self.di.north_west * 3) or d == (self.di.north_west * 4) or d == (self.di.north_west * 5) or d == (self.di.north_west * 6) or d == (self.di.north_west * 7) or d == (self.di.north_west * 8):
                        move_direction = self.l.UP_LEFT_PRO
                    elif d == self.di.north or d == (self.di.north * 2) or d == (self.di.north * 3) or d == (self.di.north * 4) or d == (self.di.north * 5) or d == (self.di.north * 6) or d == (self.di.north * 7) or d == (self.di.north * 8):
                        move_direction = self.l.UP_PRO
                    elif d == self.di.north_east or d == (self.di.north_east * 2) or d == (self.di.north_east * 3) or d == (self.di.north_east * 4) or d == (self.di.north_east * 5) or d == (self.di.north_east * 6) or d == (self.di.north_east * 7) or d == (self.di.north_east * 8):
                        move_direction = self.l.UP_RIGHT_PRO
                    elif d == self.di.east or d == (self.di.east * 2) or d == (self.di.east * 3) or d == (self.di.east * 4) or d == (self.di.east * 5) or d == (self.di.east * 6) or d == (self.di.east * 7) or d == (self.di.east * 8):
                        move_direction = self.l.RIGHT_PRO
                    elif d == self.di.south_east or d == (self.di.south_east * 2) or d == (self.di.south_east * 3) or d == (self.di.south_east * 4) or d == (self.di.south_east * 5) or d == (self.di.south_east * 6) or d == (self.di.south_east * 7) or d == (self.di.south_east * 8):
                        move_direction = self.l.DOWN_RIGHT_PRO
                    elif d == self.di.south or d == (self.di.south * 2) or d == (self.di.south * 3) or d == (self.di.south * 4) or d == (self.di.south * 5) or d == (self.di.south * 6) or d == (self.di.south * 7) or d == (self.di.south * 8):
                        move_direction = self.l.DOWN_PRO
                    elif d == self.di.south_west or d == (self.di.south_west * 2) or d == (self.di.south_west * 3) or d == (self.di.south_west * 4) or d == (self.di.south_west * 5) or d == (self.di.south_west * 6) or d == (self.di.south_west * 7) or d == (self.di.south_west * 8):
                        move_direction = self.l.DOWN_LEFT_PRO
                    elif d == self.di.west or d == (self.di.west * 2) or d == (self.di.west * 3) or d == (self.di.west * 4) or d == (self.di.west * 5) or d == (self.di.west * 6) or d == (self.di.west * 7) or d == (self.di.west * 8):
                        move_direction = self.l.LEFT_PRO
                else:
                    #桂馬を動かす手の場合
                    if d == self.di.knight_north_west:
                        move_direction = self.l.UP_LEFT_KNIGHT_PRO
                    elif d == self.di.knight_north_east:
                        move_direction = self.l.UP_RIGHT_KNIGHT_PRO
                    elif d == self.di.knight_south_east:
                        move_direction = self.l.DOWN_RIGHT_KNIGHT_PRO
                    elif d == self.di.knight_south_west:
                        move_direction = self.l.DOWN_LEFT_KNIGHT_PRO
        else:            
            #駒打ちの場合→ifromには81が入っている
            if m.piece_to_move == self.pc.pawn:
                move_direction = self.l.DROP_PAWN
            elif m.piece_to_move == self.pc.lance:
                move_direction = self.l.DROP_LANCE
            elif m.piece_to_move == self.pc.knight:
                move_direction = self.l.DROP_KNIGHT
            elif m.piece_to_move == self.pc.silver:
                move_direction = self.l.DROP_SILVER
            elif m.piece_to_move == self.pc.gold:
                move_direction = self.l.DROP_GOLD
            elif m.piece_to_move == self.pc.bishop:
                move_direction = self.l.DROP_BISHOP
            elif m.piece_to_move == self.pc.rook:
                move_direction = self.l.DROP_ROOK
        a = np.zeros(bo.nfile * bo.nrank)
        a[m.ito] = move_direction
        #move_label = a.reshape((bo.nfile, bo.nrank))
        move_label = a
        return move_label, move_direction