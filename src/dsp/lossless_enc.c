// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transform methods for lossless encoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/histogram_enc.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

// lookup table for small values of log2(int) * (1 << LOG_2_PRECISION_BITS).
// Obtained in Python with:
// a = [ str(round((1<<23)*math.log2(i))) if i else "0" for i in range(256)]
// print(',\n'.join(['  '+','.join(v)
//       for v in batched([i.rjust(9) for i in a],7)]))
const uint32_t kLog2Table[LOG_LOOKUP_IDX_MAX] = {
    0,        0,        8388608,  13295629, 16777216, 19477745, 21684237,
    23549800, 25165824, 26591258, 27866353, 29019816, 30072845, 31041538,
    31938408, 32773374, 33554432, 34288123, 34979866, 35634199, 36254961,
    36845429, 37408424, 37946388, 38461453, 38955489, 39430146, 39886887,
    40327016, 40751698, 41161982, 41558811, 41943040, 42315445, 42676731,
    43027545, 43368474, 43700062, 44022807, 44337167, 44643569, 44942404,
    45234037, 45518808, 45797032, 46069003, 46334996, 46595268, 46850061,
    47099600, 47344097, 47583753, 47818754, 48049279, 48275495, 48497560,
    48715624, 48929828, 49140306, 49347187, 49550590, 49750631, 49947419,
    50141058, 50331648, 50519283, 50704053, 50886044, 51065339, 51242017,
    51416153, 51587818, 51757082, 51924012, 52088670, 52251118, 52411415,
    52569616, 52725775, 52879946, 53032177, 53182516, 53331012, 53477707,
    53622645, 53765868, 53907416, 54047327, 54185640, 54322389, 54457611,
    54591338, 54723604, 54854440, 54983876, 55111943, 55238669, 55364082,
    55488208, 55611074, 55732705, 55853126, 55972361, 56090432, 56207362,
    56323174, 56437887, 56551524, 56664103, 56775645, 56886168, 56995691,
    57104232, 57211808, 57318436, 57424133, 57528914, 57632796, 57735795,
    57837923, 57939198, 58039632, 58139239, 58238033, 58336027, 58433234,
    58529666, 58625336, 58720256, 58814437, 58907891, 59000628, 59092661,
    59183999, 59274652, 59364632, 59453947, 59542609, 59630625, 59718006,
    59804761, 59890898, 59976426, 60061354, 60145690, 60229443, 60312620,
    60395229, 60477278, 60558775, 60639726, 60720140, 60800023, 60879382,
    60958224, 61036555, 61114383, 61191714, 61268554, 61344908, 61420785,
    61496188, 61571124, 61645600, 61719620, 61793189, 61866315, 61939001,
    62011253, 62083076, 62154476, 62225457, 62296024, 62366182, 62435935,
    62505289, 62574248, 62642816, 62710997, 62778797, 62846219, 62913267,
    62979946, 63046260, 63112212, 63177807, 63243048, 63307939, 63372484,
    63436687, 63500551, 63564080, 63627277, 63690146, 63752690, 63814912,
    63876816, 63938405, 63999682, 64060650, 64121313, 64181673, 64241734,
    64301498, 64360969, 64420148, 64479040, 64537646, 64595970, 64654014,
    64711782, 64769274, 64826495, 64883447, 64940132, 64996553, 65052711,
    65108611, 65164253, 65219641, 65274776, 65329662, 65384299, 65438691,
    65492840, 65546747, 65600416, 65653847, 65707044, 65760008, 65812741,
    65865245, 65917522, 65969575, 66021404, 66073013, 66124403, 66175575,
    66226531, 66277275, 66327806, 66378127, 66428240, 66478146, 66527847,
    66577345, 66626641, 66675737, 66724635, 66773336, 66821842, 66870154,
    66918274, 66966204, 67013944, 67061497};

// lookup table for small values of int*log2(int) * (1 << LOG_2_PRECISION_BITS).
// Obtained in Python with:
// a=[ "%d"%i if i<(1<<32) else "%dull"%i
//     for i in [ round((1<<LOG_2_PRECISION_BITS)*math.log2(i)*i) if i
//     else 0 for i in range(256)]]
// print(',\n '.join([','.join(v) for v in batched([i.rjust(15)
//                      for i in a],4)]))
const uint64_t kSLog2Table[LOG_LOOKUP_IDX_MAX] = {
    0ui64,           0ui64,           16777216ui64,    39886887ui64,
    67108864ui64,    97388723ui64,    130105423ui64,   164848600ui64,
    201326592ui64,   239321324ui64,   278663526ui64,   319217973ui64,
    360874141ui64,   403539997ui64,   447137711ui64,   491600606ui64,
    536870912ui64,   582898099ui64,   629637592ui64,   677049776ui64,
    725099212ui64,   773754010ui64,   822985323ui64,   872766924ui64,
    923074875ui64,   973887230ui64,   1025183802ui64,  1076945958ui64,
    1129156447ui64,  1181799249ui64,  1234859451ui64,  1288323135ui64,
    1342177280ui64,  1396409681ui64,  1451008871ui64,  1505964059ui64,
    1561265072ui64,  1616902301ui64,  1672866655ui64,  1729149526ui64,
    1785742744ui64,  1842638548ui64,  1899829557ui64,  1957308741ui64,
    2015069397ui64,  2073105127ui64,  2131409817ui64,  2189977618ui64,
    2248802933ui64,  2307880396ui64,  2367204859ui64,  2426771383ui64,
    2486575220ui64,  2546611805ui64,  2606876748ui64,  2667365819ui64,
    2728074942ui64,  2789000187ui64,  2850137762ui64,  2911484006ui64,
    2973035382ui64,  3034788471ui64,  3096739966ui64,  3158886666ui64,
    3221225472ui64,  3283753383ui64,  3346467489ui64,  3409364969ui64,
    3472443085ui64,  3535699182ui64,  3599130679ui64,  3662735070ui64,
    3726509920ui64,  3790452862ui64,  3854561593ui64,  3918833872ui64,
    3983267519ui64,  4047860410ui64,  4112610476ui64,  4177515704ui64,
    4242574127ui64,  4307783833ui64,  4373142952ui64,  4438649662ui64,
    4504302186ui64,  4570098787ui64,  4636037770ui64,  4702117480ui64,
    4768336298ui64,  4834692645ui64,  4901184974ui64,  4967811774ui64,
    5034571569ui64,  5101462912ui64,  5168484389ui64,  5235634615ui64,
    5302912235ui64,  5370315922ui64,  5437844376ui64,  5505496324ui64,
    5573270518ui64,  5641165737ui64,  5709180782ui64,  5777314477ui64,
    5845565671ui64,  5913933235ui64,  5982416059ui64,  6051013057ui64,
    6119723161ui64,  6188545324ui64,  6257478518ui64,  6326521733ui64,
    6395673979ui64,  6464934282ui64,  6534301685ui64,  6603775250ui64,
    6673354052ui64,  6743037185ui64,  6812823756ui64,  6882712890ui64,
    6952703725ui64,  7022795412ui64,  7092987118ui64,  7163278025ui64,
    7233667324ui64,  7304154222ui64,  7374737939ui64,  7445417707ui64,
    7516192768ui64,  7587062379ui64,  7658025806ui64,  7729082328ui64,
    7800231234ui64,  7871471825ui64,  7942803410ui64,  8014225311ui64,
    8085736859ui64,  8157337394ui64,  8229026267ui64,  8300802839ui64,
    8372666477ui64,  8444616560ui64,  8516652476ui64,  8588773618ui64,
    8660979393ui64,  8733269211ui64,  8805642493ui64,  8878098667ui64,
    8950637170ui64,  9023257446ui64,  9095958945ui64,  9168741125ui64,
    9241603454ui64,  9314545403ui64,  9387566451ui64,  9460666086ui64,
    9533843800ui64,  9607099093ui64,  9680431471ui64,  9753840445ui64,
    9827325535ui64,  9900886263ui64,  9974522161ui64,  10048232765ui64,
    10122017615ui64, 10195876260ui64, 10269808253ui64, 10343813150ui64,
    10417890516ui64, 10492039919ui64, 10566260934ui64, 10640553138ui64,
    10714916116ui64, 10789349456ui64, 10863852751ui64, 10938425600ui64,
    11013067604ui64, 11087778372ui64, 11162557513ui64, 11237404645ui64,
    11312319387ui64, 11387301364ui64, 11462350205ui64, 11537465541ui64,
    11612647010ui64, 11687894253ui64, 11763206912ui64, 11838584638ui64,
    11914027082ui64, 11989533899ui64, 12065104750ui64, 12140739296ui64,
    12216437206ui64, 12292198148ui64, 12368021795ui64, 12443907826ui64,
    12519855920ui64, 12595865759ui64, 12671937032ui64, 12748069427ui64,
    12824262637ui64, 12900516358ui64, 12976830290ui64, 13053204134ui64,
    13129637595ui64, 13206130381ui64, 13282682202ui64, 13359292772ui64,
    13435961806ui64, 13512689025ui64, 13589474149ui64, 13666316903ui64,
    13743217014ui64, 13820174211ui64, 13897188225ui64, 13974258793ui64,
    14051385649ui64, 14128568535ui64, 14205807192ui64, 14283101363ui64,
    14360450796ui64, 14437855239ui64, 14515314443ui64, 14592828162ui64,
    14670396151ui64, 14748018167ui64, 14825693972ui64, 14903423326ui64,
    14981205995ui64, 15059041743ui64, 15136930339ui64, 15214871554ui64,
    15292865160ui64, 15370910930ui64, 15449008641ui64, 15527158071ui64,
    15605359001ui64, 15683611210ui64, 15761914485ui64, 15840268608ui64,
    15918673369ui64, 15997128556ui64, 16075633960ui64, 16154189373ui64,
    16232794589ui64, 16311449405ui64, 16390153617ui64, 16468907026ui64,
    16547709431ui64, 16626560636ui64, 16705460444ui64, 16784408661ui64,
    16863405094ui64, 16942449552ui64, 17021541845ui64, 17100681785ui64};

const VP8LPrefixCode kPrefixEncodeCode[PREFIX_LOOKUP_IDX_MAX] = {
    {0, 0},  {0, 0},  {1, 0},  {2, 0},  {3, 0},  {4, 1},  {4, 1},  {5, 1},
    {5, 1},  {6, 2},  {6, 2},  {6, 2},  {6, 2},  {7, 2},  {7, 2},  {7, 2},
    {7, 2},  {8, 3},  {8, 3},  {8, 3},  {8, 3},  {8, 3},  {8, 3},  {8, 3},
    {8, 3},  {9, 3},  {9, 3},  {9, 3},  {9, 3},  {9, 3},  {9, 3},  {9, 3},
    {9, 3},  {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
    {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
    {10, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
    {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
    {11, 4}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
    {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
    {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
    {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
    {12, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
    {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
    {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
    {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
    {13, 5}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
    {14, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
    {15, 6}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
    {16, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
    {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
};

const uint8_t kPrefixEncodeExtraBitsValue[PREFIX_LOOKUP_IDX_MAX] = {
    0,   0,   0,   0,   0,   0,   1,   0,   1,   0,   1,   2,   3,   0,   1,
    2,   3,   0,   1,   2,   3,   4,   5,   6,   7,   0,   1,   2,   3,   4,
    5,   6,   7,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
    12,  13,  14,  15,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,
    11,  12,  13,  14,  15,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
    25,  26,  27,  28,  29,  30,  31,  0,   1,   2,   3,   4,   5,   6,   7,
    8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  0,   1,   2,   3,   4,   5,
    6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
    21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
    51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  0,   1,
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
    47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
    62,  63,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
    13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
    28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
    43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
    58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
    73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 0,   1,   2,   3,   4,
    5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
    20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,
    35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
    50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
    65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
    80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,
    95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 126};

static uint64_t FastSLog2Slow_C(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_WITH_CORRECTION_MAX) {
    const uint64_t orig_v = v;
    uint64_t correction;
#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)
    // use clz if available
    const uint64_t log_cnt = BitsLog2Floor(v) - 7;
    const uint32_t y = 1 << log_cnt;
    v >>= log_cnt;
#else
    uint64_t log_cnt = 0;
    uint32_t y = 1;
    do {
      ++log_cnt;
      v = v >> 1;
      y = y << 1;
    } while (v >= LOG_LOOKUP_IDX_MAX);
#endif
    // vf = (2^log_cnt) * Xf; where y = 2^log_cnt and Xf < 256
    // Xf = floor(Xf) * (1 + (v % y) / v)
    // log2(Xf) = log2(floor(Xf)) + log2(1 + (v % y) / v)
    // The correction factor: log(1 + d) ~ d; for very small d values, so
    // log2(1 + (v % y) / v) ~ LOG_2_RECIPROCAL * (v % y)/v
    correction = LOG_2_RECIPROCAL_FIXED * (orig_v & (y - 1));
    return orig_v * (kLog2Table[v] + (log_cnt << LOG_2_PRECISION_BITS)) +
           correction;
  } else {
    return (uint64_t)(LOG_2_RECIPROCAL_FIXED_DOUBLE * v * log((double)v) + .5);
  }
}

static uint32_t FastLog2Slow_C(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_WITH_CORRECTION_MAX) {
    const uint32_t orig_v = v;
    uint32_t log_2;
#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)
    // use clz if available
    const uint32_t log_cnt = BitsLog2Floor(v) - 7;
    const uint32_t y = 1 << log_cnt;
    v >>= log_cnt;
#else
    uint32_t log_cnt = 0;
    uint32_t y = 1;
    do {
      ++log_cnt;
      v = v >> 1;
      y = y << 1;
    } while (v >= LOG_LOOKUP_IDX_MAX);
#endif
    log_2 = kLog2Table[v] + (log_cnt << LOG_2_PRECISION_BITS);
    if (orig_v >= APPROX_LOG_MAX) {
      // Since the division is still expensive, add this correction factor only
      // for large values of 'v'.
      const uint64_t correction = LOG_2_RECIPROCAL_FIXED * (orig_v & (y - 1));
      log_2 += (uint32_t)DivRound(correction, orig_v);
    }
    return log_2;
  } else {
    return (uint32_t)(LOG_2_RECIPROCAL_FIXED_DOUBLE * log((double)v) + .5);
  }
}

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

// Compute the combined Shanon's entropy for distribution {X} and {X+Y}
static uint64_t CombinedShannonEntropy_C(const uint32_t X[256],
                                         const uint32_t Y[256]) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0, sumXY = 0;
  for (i = 0; i < 256; ++i) {
    const uint32_t x = X[i];
    if (x != 0) {
      const uint32_t xy = x + Y[i];
      sumX += x;
      retval += VP8LFastSLog2(x);
      sumXY += xy;
      retval += VP8LFastSLog2(xy);
    } else if (Y[i] != 0) {
      sumXY += Y[i];
      retval += VP8LFastSLog2(Y[i]);
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}

static uint64_t ShannonEntropy_C(const uint32_t* X, int n) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0;
  for (i = 0; i < n; ++i) {
    const int x = X[i];
    if (x != 0) {
      sumX += x;
      retval += VP8LFastSLog2(x);
    }
  }
  retval = VP8LFastSLog2(sumX) - retval;
  return retval;
}

void VP8LBitEntropyInit(VP8LBitEntropy* const entropy) {
  entropy->entropy = 0;
  entropy->sum = 0;
  entropy->nonzeros = 0;
  entropy->max_val = 0;
  entropy->nonzero_code = VP8L_NON_TRIVIAL_SYM;
}

void VP8LBitsEntropyUnrefined(const uint32_t* WEBP_RESTRICT const array, int n,
                              VP8LBitEntropy* WEBP_RESTRICT const entropy) {
  int i;

  VP8LBitEntropyInit(entropy);

  for (i = 0; i < n; ++i) {
    if (array[i] != 0) {
      entropy->sum += array[i];
      entropy->nonzero_code = i;
      ++entropy->nonzeros;
      entropy->entropy += VP8LFastSLog2(array[i]);
      if (entropy->max_val < array[i]) {
        entropy->max_val = array[i];
      }
    }
  }
  entropy->entropy = VP8LFastSLog2(entropy->sum) - entropy->entropy;
}

static WEBP_INLINE void GetEntropyUnrefinedHelper(
    uint32_t val, int i, uint32_t* WEBP_RESTRICT const val_prev,
    int* WEBP_RESTRICT const i_prev,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  const int streak = i - *i_prev;

  // Gather info for the bit entropy.
  if (*val_prev != 0) {
    bit_entropy->sum += (*val_prev) * streak;
    bit_entropy->nonzeros += streak;
    bit_entropy->nonzero_code = *i_prev;
    bit_entropy->entropy += VP8LFastSLog2(*val_prev) * streak;
    if (bit_entropy->max_val < *val_prev) {
      bit_entropy->max_val = *val_prev;
    }
  }

  // Gather info for the Huffman cost.
  stats->counts[*val_prev != 0] += (streak > 3);
  stats->streaks[*val_prev != 0][(streak > 3)] += streak;

  *val_prev = val;
  *i_prev = i;
}

static void GetEntropyUnrefined_C(
    const uint32_t X[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i;
  int i_prev = 0;
  uint32_t x_prev = X[0];

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);

  for (i = 1; i < length; ++i) {
    const uint32_t x = X[i];
    if (x != x_prev) {
      GetEntropyUnrefinedHelper(x, i, &x_prev, &i_prev, bit_entropy, stats);
    }
  }
  GetEntropyUnrefinedHelper(0, i, &x_prev, &i_prev, bit_entropy, stats);

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

static void GetCombinedEntropyUnrefined_C(
    const uint32_t X[], const uint32_t Y[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i = 1;
  int i_prev = 0;
  uint32_t xy_prev = X[0] + Y[0];

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);

  for (i = 1; i < length; ++i) {
    const uint32_t xy = X[i] + Y[i];
    if (xy != xy_prev) {
      GetEntropyUnrefinedHelper(xy, i, &xy_prev, &i_prev, bit_entropy, stats);
    }
  }
  GetEntropyUnrefinedHelper(0, i, &xy_prev, &i_prev, bit_entropy, stats);

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

//------------------------------------------------------------------------------

void VP8LSubtractGreenFromBlueAndRed_C(uint32_t* argb_data, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const int argb = (int)argb_data[i];
    const int green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = (((argb >> 0) & 0xff) - green) & 0xff;
    argb_data[i] = ((uint32_t)argb & 0xff00ff00u) | (new_r << 16) | new_b;
  }
}

static WEBP_INLINE int ColorTransformDelta(int8_t color_pred, int8_t color) {
  return ((int)color_pred * color) >> 5;
}

static WEBP_INLINE int8_t U32ToS8(uint32_t v) { return (int8_t)(v & 0xff); }

void VP8LTransformColor_C(const VP8LMultipliers* WEBP_RESTRICT const m,
                          uint32_t* WEBP_RESTRICT data, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const int8_t green = U32ToS8(argb >> 8);
    const int8_t red = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta((int8_t)m->green_to_red, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta((int8_t)m->green_to_blue, green);
    new_blue -= ColorTransformDelta((int8_t)m->red_to_blue, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

static WEBP_INLINE uint8_t TransformColorRed(uint8_t green_to_red,
                                             uint32_t argb) {
  const int8_t green = U32ToS8(argb >> 8);
  int new_red = argb >> 16;
  new_red -= ColorTransformDelta((int8_t)green_to_red, green);
  return (new_red & 0xff);
}

static WEBP_INLINE uint8_t TransformColorBlue(uint8_t green_to_blue,
                                              uint8_t red_to_blue,
                                              uint32_t argb) {
  const int8_t green = U32ToS8(argb >> 8);
  const int8_t red = U32ToS8(argb >> 16);
  int new_blue = argb & 0xff;
  new_blue -= ColorTransformDelta((int8_t)green_to_blue, green);
  new_blue -= ColorTransformDelta((int8_t)red_to_blue, red);
  return (new_blue & 0xff);
}

void VP8LCollectColorRedTransforms_C(const uint32_t* WEBP_RESTRICT argb,
                                     int stride, int tile_width,
                                     int tile_height, int green_to_red,
                                     uint32_t histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
    }
    argb += stride;
  }
}

void VP8LCollectColorBlueTransforms_C(const uint32_t* WEBP_RESTRICT argb,
                                      int stride, int tile_width,
                                      int tile_height, int green_to_blue,
                                      int red_to_blue, uint32_t histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorBlue((uint8_t)green_to_blue, (uint8_t)red_to_blue,
                                 argb[x])];
    }
    argb += stride;
  }
}

//------------------------------------------------------------------------------

static int VectorMismatch_C(const uint32_t* const array1,
                            const uint32_t* const array2, int length) {
  int match_len = 0;

  while (match_len < length && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
}

// Bundles multiple (1, 2, 4 or 8) pixels into a single pixel.
void VP8LBundleColorMap_C(const uint8_t* WEBP_RESTRICT const row, int width,
                          int xbits, uint32_t* WEBP_RESTRICT dst) {
  int x;
  if (xbits > 0) {
    const int bit_depth = 1 << (3 - xbits);
    const int mask = (1 << xbits) - 1;
    uint32_t code = 0xff000000;
    for (x = 0; x < width; ++x) {
      const int xsub = x & mask;
      if (xsub == 0) {
        code = 0xff000000;
      }
      code |= row[x] << (8 + bit_depth * xsub);
      dst[x >> xbits] = code;
    }
  } else {
    for (x = 0; x < width; ++x) dst[x] = 0xff000000 | (row[x] << 8);
  }
}

//------------------------------------------------------------------------------

static uint32_t ExtraCost_C(const uint32_t* population, int length) {
  int i;
  uint32_t cost = population[4] + population[5];
  assert(length % 2 == 0);
  for (i = 2; i < length / 2 - 1; ++i) {
    cost += i * (population[2 * i + 2] + population[2 * i + 3]);
  }
  return cost;
}

//------------------------------------------------------------------------------

static void AddVector_C(const uint32_t* WEBP_RESTRICT a,
                        const uint32_t* WEBP_RESTRICT b,
                        uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

static void AddVectorEq_C(const uint32_t* WEBP_RESTRICT a,
                          uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i < size; ++i) out[i] += a[i];
}

//------------------------------------------------------------------------------
// Image transforms.

static void PredictorSub0_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i < num_pixels; ++i) out[i] = VP8LSubPixels(in[i], ARGB_BLACK);
  (void)upper;
}

static void PredictorSub1_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i < num_pixels; ++i) out[i] = VP8LSubPixels(in[i], in[i - 1]);
  (void)upper;
}

// It subtracts the prediction from the input pixel and stores the residual
// in the output pixel.
#define GENERATE_PREDICTOR_SUB(PREDICTOR_I)                      \
  static void PredictorSub##PREDICTOR_I##_C(                     \
      const uint32_t* in, const uint32_t* upper, int num_pixels, \
      uint32_t* WEBP_RESTRICT out) {                             \
    int x;                                                       \
    assert(upper != NULL);                                       \
    for (x = 0; x < num_pixels; ++x) {                           \
      const uint32_t pred =                                      \
          VP8LPredictor##PREDICTOR_I##_C(&in[x - 1], upper + x); \
      out[x] = VP8LSubPixels(in[x], pred);                       \
    }                                                            \
  }

GENERATE_PREDICTOR_SUB(2)
GENERATE_PREDICTOR_SUB(3)
GENERATE_PREDICTOR_SUB(4)
GENERATE_PREDICTOR_SUB(5)
GENERATE_PREDICTOR_SUB(6)
GENERATE_PREDICTOR_SUB(7)
GENERATE_PREDICTOR_SUB(8)
GENERATE_PREDICTOR_SUB(9)
GENERATE_PREDICTOR_SUB(10)
GENERATE_PREDICTOR_SUB(11)
GENERATE_PREDICTOR_SUB(12)
GENERATE_PREDICTOR_SUB(13)

//------------------------------------------------------------------------------

VP8LProcessEncBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;
VP8LProcessEncBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed_SSE;

VP8LTransformColorFunc VP8LTransformColor;
VP8LTransformColorFunc VP8LTransformColor_SSE;

VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms;
VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms_SSE;
VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms;
VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms_SSE;

VP8LFastLog2SlowFunc VP8LFastLog2Slow;
VP8LFastSLog2SlowFunc VP8LFastSLog2Slow;

VP8LCostFunc VP8LExtraCost;
VP8LCombinedShannonEntropyFunc VP8LCombinedShannonEntropy;
VP8LShannonEntropyFunc VP8LShannonEntropy;

VP8LGetEntropyUnrefinedFunc VP8LGetEntropyUnrefined;
VP8LGetCombinedEntropyUnrefinedFunc VP8LGetCombinedEntropyUnrefined;

VP8LAddVectorFunc VP8LAddVector;
VP8LAddVectorEqFunc VP8LAddVectorEq;

VP8LVectorMismatchFunc VP8LVectorMismatch;
VP8LBundleColorMapFunc VP8LBundleColorMap;
VP8LBundleColorMapFunc VP8LBundleColorMap_SSE;

VP8LPredictorAddSubFunc VP8LPredictorsSub[16];
VP8LPredictorAddSubFunc VP8LPredictorsSub_C[16];
VP8LPredictorAddSubFunc VP8LPredictorsSub_SSE[16];

extern VP8CPUInfo VP8GetCPUInfo;
extern void VP8LEncDspInitSSE2(void);
extern void VP8LEncDspInitSSE41(void);
extern void VP8LEncDspInitAVX2(void);
extern void VP8LEncDspInitNEON(void);
extern void VP8LEncDspInitMIPS32(void);
extern void VP8LEncDspInitMIPSdspR2(void);
extern void VP8LEncDspInitMSA(void);

WEBP_DSP_INIT_FUNC(VP8LEncDspInit) {
  VP8LDspInit();

#if !WEBP_NEON_OMIT_C_CODE
  VP8LSubtractGreenFromBlueAndRed = VP8LSubtractGreenFromBlueAndRed_C;

  VP8LTransformColor = VP8LTransformColor_C;
#endif

  VP8LCollectColorBlueTransforms = VP8LCollectColorBlueTransforms_C;
  VP8LCollectColorRedTransforms = VP8LCollectColorRedTransforms_C;

  VP8LFastLog2Slow = FastLog2Slow_C;
  VP8LFastSLog2Slow = FastSLog2Slow_C;

  VP8LExtraCost = ExtraCost_C;
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_C;
  VP8LShannonEntropy = ShannonEntropy_C;

  VP8LGetEntropyUnrefined = GetEntropyUnrefined_C;
  VP8LGetCombinedEntropyUnrefined = GetCombinedEntropyUnrefined_C;

  VP8LAddVector = AddVector_C;
  VP8LAddVectorEq = AddVectorEq_C;

  VP8LVectorMismatch = VectorMismatch_C;
  VP8LBundleColorMap = VP8LBundleColorMap_C;

  VP8LPredictorsSub[0] = PredictorSub0_C;
  VP8LPredictorsSub[1] = PredictorSub1_C;
  VP8LPredictorsSub[2] = PredictorSub2_C;
  VP8LPredictorsSub[3] = PredictorSub3_C;
  VP8LPredictorsSub[4] = PredictorSub4_C;
  VP8LPredictorsSub[5] = PredictorSub5_C;
  VP8LPredictorsSub[6] = PredictorSub6_C;
  VP8LPredictorsSub[7] = PredictorSub7_C;
  VP8LPredictorsSub[8] = PredictorSub8_C;
  VP8LPredictorsSub[9] = PredictorSub9_C;
  VP8LPredictorsSub[10] = PredictorSub10_C;
  VP8LPredictorsSub[11] = PredictorSub11_C;
  VP8LPredictorsSub[12] = PredictorSub12_C;
  VP8LPredictorsSub[13] = PredictorSub13_C;
  VP8LPredictorsSub[14] = PredictorSub0_C;  // <- padding security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_C;

  VP8LPredictorsSub_C[0] = PredictorSub0_C;
  VP8LPredictorsSub_C[1] = PredictorSub1_C;
  VP8LPredictorsSub_C[2] = PredictorSub2_C;
  VP8LPredictorsSub_C[3] = PredictorSub3_C;
  VP8LPredictorsSub_C[4] = PredictorSub4_C;
  VP8LPredictorsSub_C[5] = PredictorSub5_C;
  VP8LPredictorsSub_C[6] = PredictorSub6_C;
  VP8LPredictorsSub_C[7] = PredictorSub7_C;
  VP8LPredictorsSub_C[8] = PredictorSub8_C;
  VP8LPredictorsSub_C[9] = PredictorSub9_C;
  VP8LPredictorsSub_C[10] = PredictorSub10_C;
  VP8LPredictorsSub_C[11] = PredictorSub11_C;
  VP8LPredictorsSub_C[12] = PredictorSub12_C;
  VP8LPredictorsSub_C[13] = PredictorSub13_C;
  VP8LPredictorsSub_C[14] = PredictorSub0_C;  // <- padding security sentinels
  VP8LPredictorsSub_C[15] = PredictorSub0_C;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_HAVE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8LEncDspInitSSE2();
#if defined(WEBP_HAVE_SSE41)
      if (VP8GetCPUInfo(kSSE4_1)) {
        VP8LEncDspInitSSE41();
#if defined(WEBP_HAVE_AVX2)
        if (VP8GetCPUInfo(kAVX2)) {
          VP8LEncDspInitAVX2();
        }
#endif
      }
#endif
    }
#endif
#if defined(WEBP_USE_MIPS32)
    if (VP8GetCPUInfo(kMIPS32)) {
      VP8LEncDspInitMIPS32();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8LEncDspInitMIPSdspR2();
    }
#endif
#if defined(WEBP_USE_MSA)
    if (VP8GetCPUInfo(kMSA)) {
      VP8LEncDspInitMSA();
    }
#endif
  }

#if defined(WEBP_HAVE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (VP8GetCPUInfo != NULL && VP8GetCPUInfo(kNEON))) {
    VP8LEncDspInitNEON();
  }
#endif

  assert(VP8LSubtractGreenFromBlueAndRed != NULL);
  assert(VP8LTransformColor != NULL);
  assert(VP8LCollectColorBlueTransforms != NULL);
  assert(VP8LCollectColorRedTransforms != NULL);
  assert(VP8LFastLog2Slow != NULL);
  assert(VP8LFastSLog2Slow != NULL);
  assert(VP8LExtraCost != NULL);
  assert(VP8LCombinedShannonEntropy != NULL);
  assert(VP8LShannonEntropy != NULL);
  assert(VP8LGetEntropyUnrefined != NULL);
  assert(VP8LGetCombinedEntropyUnrefined != NULL);
  assert(VP8LAddVector != NULL);
  assert(VP8LAddVectorEq != NULL);
  assert(VP8LVectorMismatch != NULL);
  assert(VP8LBundleColorMap != NULL);
  assert(VP8LPredictorsSub[0] != NULL);
  assert(VP8LPredictorsSub[1] != NULL);
  assert(VP8LPredictorsSub[2] != NULL);
  assert(VP8LPredictorsSub[3] != NULL);
  assert(VP8LPredictorsSub[4] != NULL);
  assert(VP8LPredictorsSub[5] != NULL);
  assert(VP8LPredictorsSub[6] != NULL);
  assert(VP8LPredictorsSub[7] != NULL);
  assert(VP8LPredictorsSub[8] != NULL);
  assert(VP8LPredictorsSub[9] != NULL);
  assert(VP8LPredictorsSub[10] != NULL);
  assert(VP8LPredictorsSub[11] != NULL);
  assert(VP8LPredictorsSub[12] != NULL);
  assert(VP8LPredictorsSub[13] != NULL);
  assert(VP8LPredictorsSub[14] != NULL);
  assert(VP8LPredictorsSub[15] != NULL);
  assert(VP8LPredictorsSub_C[0] != NULL);
  assert(VP8LPredictorsSub_C[1] != NULL);
  assert(VP8LPredictorsSub_C[2] != NULL);
  assert(VP8LPredictorsSub_C[3] != NULL);
  assert(VP8LPredictorsSub_C[4] != NULL);
  assert(VP8LPredictorsSub_C[5] != NULL);
  assert(VP8LPredictorsSub_C[6] != NULL);
  assert(VP8LPredictorsSub_C[7] != NULL);
  assert(VP8LPredictorsSub_C[8] != NULL);
  assert(VP8LPredictorsSub_C[9] != NULL);
  assert(VP8LPredictorsSub_C[10] != NULL);
  assert(VP8LPredictorsSub_C[11] != NULL);
  assert(VP8LPredictorsSub_C[12] != NULL);
  assert(VP8LPredictorsSub_C[13] != NULL);
  assert(VP8LPredictorsSub_C[14] != NULL);
  assert(VP8LPredictorsSub_C[15] != NULL);
}

//------------------------------------------------------------------------------
