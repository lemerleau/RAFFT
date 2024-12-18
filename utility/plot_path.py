#!/usr/bin/env python3
"""Take the output of rafft and produce a latex file to display the fold paths.

It uses varna to produce 2ndary structures.

XXX 2024-12-07: With fixes from @SchrammAntoine
"""

import argparse
import subprocess
from os import system
from os.path import realpath, dirname
from PIL import Image, ImageDraw, ImageFont
import aggdraw
from colour import Color


def paired_positions(structure):
    "return a list of pairs (paired positions)"
    # save open bracket in piles
    pile_reg, pile_pk = [], []
    pairs = []
    for i, sstruc in enumerate(structure):
        if sstruc in ["<", "("]:
            pile_reg += [i]
        elif sstruc == "[":
            pile_pk += [i]
        elif sstruc in [">", ")"]:
            pairs += [(pile_reg.pop(), i)]
        elif sstruc == "]":
            pairs += [(pile_pk.pop(), i)]
    return pairs


def get_gradient(quant):
    # blue = Color("LightSkyBlue")
    blue = Color("LightCyan")
    colors = list(blue.range_to(Color("MidnightBlue"), 40))
    col = min(int(quant * len(colors)), len(colors)-1)
    color = tuple([int(el * 255) for el in colors[col].rgb])
    return color


def get_gradient_img():
    img = Image.new('RGBA', (100, 900))
    for i in range(900):
        col = get_gradient(float(i)/900)
        for j in range(100):
            img.putpixel((j, i), col)
    return img


def add_grad_img(path_img, max_val, min_val, fnt):
    "add the logo gradient image"
    grad_img = get_gradient_img()
    grad_x = int(path_img.size[0] * 0.15)
    grad_y = int(path_img.size[1] * 0.02)
    grad_img = grad_img.resize((grad_y, grad_x))
    path_img.paste(grad_img, (0, 0))

    draw = ImageDraw.Draw(path_img)
    #x_size, y_size = fnt.getsize("{:.1f}".format(max_val)) ## DEPRECATED SchrammAntoine
    bbox = fnt.getbbox("{:.1f}".format(max_val)) # update
    x_size, y_size = bbox[2] - bbox[0], bbox[3] - bbox[1] # update
    draw.text((grad_y, 0), "{:.2f}".format(max_val), "black", fnt)
    draw.text((grad_y, grad_x-y_size), "{:.2f}".format(min_val), "black", fnt)
    draw = ImageDraw.Draw(path_img)


def parse_rafft_output(infile):
    results = []
    with open(infile) as rafft_out:
        seq = rafft_out.readline().strip()
        for l in rafft_out:
            if l.startswith("# --"):
                results += [[]]
            else:
                struct, nrj = l.strip().split()
                results[-1] += [(struct, float(nrj))]
    return results, seq


def get_connected_prev(cur_struct, prev_pos):
    "get the connected structures"
    cur_pairs = set(paired_positions(cur_struct))
    res = []
    for si, (struct, nrj) in enumerate(prev_pos):
        pairs = set(paired_positions(struct))
        if len(pairs - cur_pairs) == 0:
            res += [si]
    return res


def parse_arguments():
    """Parsing command line
    """
    parser = argparse.ArgumentParser(
        description="Uses VARNA to plot the fast-paths predicted by RAFFT. !! It creates a temporary directory in the current folder!!")
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
    parser.add_argument(
        '--width', '-wi', help="figure width", type=int, default=500)
    parser.add_argument('--height', '-he',
                        help="figure height", type=int, default=300)
    parser.add_argument('--res_varna', '-rv',
                        help="change varna resolution", type=float, default=1.0)
    parser.add_argument('--line_thick', '-lt',
                        help="line thickness", type=float, default=1)
    parser.add_argument('--font_size', '-fs',
                        help="font size for the colors", type=int, default=20)
    parser.add_argument(
        '--varna_jar', help="varna jar (please download it from VARNA website)")
    parser.add_argument('--no_col', action="store_true",
                        help="don't use the color gradient for the edges")
    parser.add_argument('--no_fig', action="store_true",
                        help="you already computed the structures previously?")
    return parser.parse_args()


def main():
    args = parse_arguments()

    fast_paths, seq = parse_rafft_output(args.rafft_out)

    # draw structures
    out_dir = "./tmp_rafft_fig"
    system(f"mkdir -p {out_dir}")
    varna_jar = "{}/VARNAv3-93.jar".format(
        dirname(realpath(__file__))) if args.varna_jar is None else args.varna_jar
    varna = f"{varna_jar} fr.orsay.lri.varna.applications.VARNAcmd"
    cmd_line = "java -cp {}  -sequenceDBN '{}' -structureDBN '{}' -o {} -resolution {} -algorithm naview -bpStyle 'simple' -fillBases True -spaceBetweenBases 0.5 -baseInner '#051C2C' -baseName '#FFFFFF00' -baseName '#051C2C' -background '#00000000' -periodNum 1000 -baseNum '#00FFFFFF' 2>&1 1> /dev/null"

    if not args.no_fig:
        for step_i, fold_step in enumerate(fast_paths):
            for str_i, (struct, nrj) in enumerate(fold_step):
                out_file = f"{out_dir}/s{step_i}_{str_i}.png"
                cmd_line_c = cmd_line.format(
                    varna, " "*len(seq), struct, out_file, args.res_varna)
                subprocess.Popen(
                    cmd_line_c, stdout=subprocess.PIPE, shell=True).communicate()

    width, height = args.width, args.height
    canvas = (width, height)
    path_img = Image.new('RGBA', canvas, 'white')
    # to draw the paths
    nb_steps = len(fast_paths)
    nb_saved = max((len(el) for el in fast_paths))
    rate_w, rate_h = float(
        width)/float(nb_steps), float(height)/float(nb_saved)

    # save position in the canvas for each structure
    actual_position, actual_sizes = {}, {}

    # save nrj differences
    nrj_changes = {}

    # width of points
    pw = args.line_thick
    pos_hor = 0
    crop_side = 0
    # store best change
    min_change = 0

    for step_i, fold_step in enumerate(fast_paths):
        pos_vert = 0
        tmp_left = 0
        if len(fold_step) > 1:
            # store resize rates
            for str_i, (struct, nrj) in enumerate(fold_step):
                out_file = f"{out_dir}/s{step_i}_{str_i}.png"
                cur_str = Image.open(out_file)
                fig_w, fig_h = cur_str.size
                if fig_w < fig_h:
                    cur_str = cur_str.transpose(Image.ROTATE_90)

                fig_w, fig_h = cur_str.size
                resize_rate = min(rate_w/float(fig_w), rate_h/float(fig_h))

                n_fig_w, n_fig_h = int(
                    fig_w * resize_rate), int(fig_h * resize_rate)

                #cur_str = cur_str.resize((n_fig_w, n_fig_h), Image.ANTIALIAS) ## DEPRECATED SchrammAntoine
                cur_str = cur_str.resize((n_fig_w, n_fig_h), Image.Resampling.LANCZOS) # update

                actual_sizes[(step_i, str_i)] = (n_fig_w, n_fig_h)
                actual_position[(step_i, str_i)] = (
                    pos_hor+n_fig_w, pos_vert+n_fig_h//2)

                lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
                nrj_changes[(step_i, str_i)] = {}

                for si in lprev_co:
                    prev_st, prev_nrj = fast_paths[step_i-1][si]
                    nrj_changes[(step_i, str_i)][(
                        step_i-1, si)] = nrj - prev_nrj

                    if nrj_changes[(step_i, str_i)][(step_i-1, si)] <= min_change:
                        min_change = nrj_changes[(
                            step_i, str_i)][(step_i-1, si)]

                path_img.paste(cur_str, (pos_hor, pos_vert))
                pos_vert += int(n_fig_h)

                if n_fig_w > tmp_left:
                    tmp_left = n_fig_w
        else:
            str_i = 0
            out_file = f"{out_dir}/s{step_i}_{str_i}.png"
            cur_str = Image.open(out_file)
            fig_w, fig_h = cur_str.size
            resize_rate = min(rate_w/float(fig_w), rate_h/float(fig_h))
            n_fig_w, n_fig_h = int(
                fig_w * resize_rate), int(fig_h * resize_rate)
            #cur_str = cur_str.resize((n_fig_w, n_fig_h), Image.ANTIALIA S) ## DEPRECATED SchrammAntoine
            cur_str = cur_str.resize((n_fig_w, n_fig_h),Image.Resampling.LANCZOS) # update
            path_img.paste(cur_str, (pos_hor, height//2 - n_fig_h//2))
            actual_position[(step_i, str_i)] = (pos_hor + n_fig_w, height//2)
            tmp_left = n_fig_w

        if pos_vert > height:
            print("error", step_i, pos_vert)

        crop_side = pos_hor + tmp_left
        pos_hor += max(int(tmp_left), int(rate_w))

    # past the paths
    for step_i, fold_step in enumerate(fast_paths):
        if len(fold_step) > 1:
            for str_i, (struct, nrj) in enumerate(fold_step):
                lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])
                draw_path = aggdraw.Draw(path_img)
                n_fig_w, n_fig_h = actual_sizes[(step_i, str_i)]

                for si in lprev_co:
                    prev_w, prev_h = actual_position[(step_i-1, si)]
                    cur_w, cur_h = actual_position[(step_i, str_i)]
                    nrj_sta = nrj_changes[(step_i, str_i)][(step_i-1, si)]
                    cur_w -= n_fig_w

                    pathstring = " M{},{} C{},{},{},{},{},{}".format(int(prev_w), int(prev_h),
                                                                     int(prev_w + (cur_w -
                                                                         prev_w)//2), int(prev_h),
                                                                     int(prev_w + (cur_w -
                                                                         prev_w)//2), int(cur_h),
                                                                     int(cur_w), int(cur_h))

                    symbol = aggdraw.Symbol(pathstring)
                    if args.no_col:
                        outline = aggdraw.Pen("black", pw)
                    else:
                        outline = aggdraw.Pen(
                            get_gradient(nrj_sta/min_change), pw)
                    draw_path.symbol((0, 0), symbol, outline)
                    draw_path.flush()

    fnt_pos = ImageFont.truetype(
        "{}/Times_New_Roman.ttf".format(dirname(realpath(__file__))), args.font_size)
    path_img = path_img.crop((0, 0, crop_side, height))

    if not args.no_col:
        add_grad_img(path_img, 0, min_change, fnt_pos)

    if args.out is not None:
        path_img.save(args.out)
    else:
        path_img.show()


if __name__ == '__main__':
    main()
