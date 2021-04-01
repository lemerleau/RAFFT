"""Take the output of rafft and produce a latex file to display the fold paths.

It uses varna to produce 2ndary structures.
"""

import argparse
import subprocess
from os import system
from os.path import realpath, dirname
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils import paired_positions
import aggdraw


def parse_rafft_outpumt(infile):
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
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('rafft_out', help="rafft_output")
    parser.add_argument('--out', '-o', help="output file")
    parser.add_argument('--width', '-wi', help="figure width", type=int, default=500)
    parser.add_argument('--height', '-he', help="figure height", type=int, default=300)
    parser.add_argument('--res_varna', '-rv', help="change varna resolution", type=float, default=1.0)
    parser.add_argument('--line_thick', '-lt', help="line thickness", type=int, default=2)
    parser.add_argument('--varna_jar', help="varna jar (please download it from VARNA website)")
    parser.add_argument('--no_fig', action="store_true", help="you already computed the structures previously?")
    return parser.parse_args()


def main():
    args = parse_arguments()

    fast_paths, seq = parse_rafft_outpumt(args.rafft_out)

    # draw structures
    out_dir = "./tmp_rafft_fig"
    system(f"mkdir -p {out_dir}")
    varna_jar = "{}/VARNAv3-93.jar".format(dirname(realpath(__file__))) if args.varna_jar is None else args.varna_jar
    varna = f"{varna_jar} fr.orsay.lri.varna.applications.VARNAcmd"
    cmd_line = "java -cp {}  -sequenceDBN {} -structureDBN '{}' -o {} -resolution {} -algorithm naview -bpStyle 'line' -fillBases True -spaceBetweenBases 0.5 -baseInner '#051C2C' -baseName '#051C2C' -background '#000000' -periodNum 1000 2>&1 1> /dev/null"

    if not args.no_fig:
        for step_i, fold_step in enumerate(fast_paths):
            for str_i, (struct, nrj) in enumerate(fold_step):
                out_file = f"{out_dir}/s{step_i}_{str_i}.png"
                cmd_line_c = cmd_line.format(varna, seq, struct, out_file, args.res_varna)
                subprocess.Popen(cmd_line_c, stdout=subprocess.PIPE, shell=True).communicate()

    width, height = args.width, args.height
    canvas = (width, height)
    path_img = Image.new('RGBA', canvas, 'white')
    # to draw the paths
    nb_steps = len(fast_paths)
    nb_saved = len(fast_paths[-1])
    rate_w, rate_h = float(width)/float(nb_steps), float(height)/float(nb_saved)

    # save position in the canvas for each structure
    actual_position = {}

    # width of points
    pw = args.line_thick
    outline = aggdraw.Pen("black", pw)
    pos_hor = 0
    for step_i, fold_step in enumerate(fast_paths):
        pos_vert = 0
        tmp_left = 0
        if len(fold_step) > 1:
            for str_i, (struct, nrj) in enumerate(fold_step):
                out_file = f"{out_dir}/s{step_i}_{str_i}.png"
                cur_str = Image.open(out_file)
                fig_w, fig_h = cur_str.size
                resize_rate = min(rate_w/float(fig_w), rate_h/float(fig_h))

                n_fig_w, n_fig_h = int(fig_w * resize_rate), int(fig_h * resize_rate)

                cur_str = cur_str.resize((n_fig_w, n_fig_h), Image.ANTIALIAS)

                actual_position[(step_i, str_i)] = (pos_hor+n_fig_w, pos_vert+n_fig_h//2)

                lprev_co = get_connected_prev(struct, fast_paths[step_i - 1])

                draw_path = aggdraw.Draw(path_img)
                for si in lprev_co:
                    prev_w, prev_h = actual_position[(step_i-1, si)]
                    cur_w, cur_h = actual_position[(step_i, str_i)]
                    cur_w -= n_fig_w

                    pathstring = " M{},{} C{},{},{},{},{},{}".format(int(prev_w), int(prev_h),
                                                                     int(prev_w + (cur_w - prev_w)//2), int(prev_h),
                                                                     int(prev_w + (cur_w - prev_w)//2), int(cur_h),
                                                                     int(cur_w), int(cur_h))
                    symbol = aggdraw.Symbol(pathstring)
                    draw_path.symbol((0, 0), symbol, outline)
                    draw_path.flush()

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
            n_fig_w, n_fig_h = int(fig_w * resize_rate), int(fig_h * resize_rate)
            cur_str = cur_str.resize((n_fig_w, n_fig_h), Image.ANTIALIAS)
            path_img.paste(cur_str, (pos_hor, height//2 - n_fig_h//2))
            actual_position[(step_i, str_i)] = (pos_hor + n_fig_w, height//2)
            tmp_left = n_fig_w

        pos_hor += max(int(tmp_left), int(rate_w))


    if args.out is not None:
        path_img.save(args.out)
    else:
        path_img.show()


if __name__ == '__main__':
    main()
