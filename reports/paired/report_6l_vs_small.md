# Paired evaluation: 6L vs small
1138 paired utterances (0 unmatched dropped)

## all (n=1138)

| metric | 6L | small | diff (small−6L) | 95 % CI | small better / tie / worse |
|---|---|---|---|---|---|
| wer | 0.0414 | 0.0399 | -0.0015 | [-0.0066, +0.0038] | 106 / 909 / 123 |
| cer | 0.0097 | 0.0078 | -0.0018 | [-0.0045, +0.0004] | 117 / 879 / 142 |
| wer_conformer | 0.0271 | 0.0240 | -0.0031 | [-0.0071, +0.0007] | 58 / 1013 / 67 |
| cer_conformer | 0.0075 | 0.0052 | -0.0022 ** | [-0.0047, -0.0001] | 58 / 995 / 85 |
| utmos | 3.4981 | 3.4729 | -0.0252 ** | [-0.0375, -0.0131] | 521 / 1 / 616 |
| spk_sim | 0.9580 | 0.9539 | -0.0041 ** | [-0.0051, -0.0031] | 450 / 32 / 656 |

failures (WER > 0.5): 6L 6 ({'other': 5, 'prompt_continuation': 1}), small 3 ({'other': 3})

## main set (n=800)

| metric | 6L | small | diff (small−6L) | 95 % CI | small better / tie / worse |
|---|---|---|---|---|---|
| wer | 0.0347 | 0.0334 | -0.0013 | [-0.0060, +0.0036] | 68 / 657 / 75 |
| cer | 0.0064 | 0.0059 | -0.0006 | [-0.0023, +0.0010] | 72 / 645 / 83 |
| wer_conformer | 0.0194 | 0.0192 | -0.0002 | [-0.0037, +0.0029] | 30 / 728 / 42 |
| cer_conformer | 0.0047 | 0.0041 | -0.0007 | [-0.0023, +0.0007] | 34 / 714 / 52 |
| utmos | 3.4978 | 3.4671 | -0.0307 ** | [-0.0444, -0.0163] | 357 / 1 / 442 |
| spk_sim | 0.9580 | 0.9536 | -0.0045 ** | [-0.0057, -0.0033] | 308 / 20 / 472 |

failures (WER > 0.5): 6L 3 ({'other': 3}), small 1 ({'other': 1})

## hard set (n=338)

| metric | 6L | small | diff (small−6L) | 95 % CI | small better / tie / worse |
|---|---|---|---|---|---|
| wer | 0.0572 | 0.0552 | -0.0020 | [-0.0150, +0.0104] | 38 / 252 / 48 |
| cer | 0.0174 | 0.0125 | -0.0049 | [-0.0125, +0.0017] | 45 / 234 / 59 |
| wer_conformer | 0.0452 | 0.0352 | -0.0100 | [-0.0214, +0.0001] | 28 / 285 / 25 |
| cer_conformer | 0.0139 | 0.0080 | -0.0059 | [-0.0139, +0.0007] | 24 / 281 / 33 |
| utmos | 3.4990 | 3.4867 | -0.0123 | [-0.0352, +0.0113] | 164 / 0 / 174 |
| spk_sim | 0.9579 | 0.9547 | -0.0032 ** | [-0.0051, -0.0012] | 142 / 12 / 184 |

failures (WER > 0.5): 6L 3 ({'prompt_continuation': 1, 'other': 2}), small 2 ({'other': 2})

## named voices (n=569)

| metric | 6L | small | diff (small−6L) | 95 % CI | small better / tie / worse |
|---|---|---|---|---|---|
| wer | 0.0446 | 0.0455 | +0.0009 | [-0.0071, +0.0089] | 54 / 439 / 76 |
| cer | 0.0100 | 0.0091 | -0.0009 | [-0.0046, +0.0023] | 62 / 423 / 84 |
| wer_conformer | 0.0281 | 0.0261 | -0.0021 | [-0.0082, +0.0035] | 29 / 501 / 39 |
| cer_conformer | 0.0075 | 0.0061 | -0.0015 | [-0.0053, +0.0016] | 31 / 489 / 49 |
| utmos | 3.4919 | 3.4697 | -0.0222 ** | [-0.0385, -0.0063] | 257 / 1 / 311 |
| spk_sim | 0.9590 | 0.9576 | -0.0013 | [-0.0027, +0.0000] | 244 / 19 / 306 |

failures (WER > 0.5): 6L 4 ({'other': 3, 'prompt_continuation': 1}), small 3 ({'other': 3})

## held-out voices (n=569)

| metric | 6L | small | diff (small−6L) | 95 % CI | small better / tie / worse |
|---|---|---|---|---|---|
| wer | 0.0381 | 0.0342 | -0.0039 | [-0.0106, +0.0026] | 52 / 470 / 47 |
| cer | 0.0093 | 0.0066 | -0.0028 | [-0.0065, +0.0004] | 55 / 456 / 58 |
| wer_conformer | 0.0260 | 0.0218 | -0.0042 | [-0.0096, +0.0007] | 29 / 512 / 28 |
| cer_conformer | 0.0074 | 0.0044 | -0.0030 | [-0.0066, +0.0001] | 27 / 506 / 36 |
| utmos | 3.5043 | 3.4761 | -0.0282 ** | [-0.0461, -0.0094] | 264 / 0 / 305 |
| spk_sim | 0.9570 | 0.9502 | -0.0068 ** | [-0.0084, -0.0053] | 206 / 13 / 350 |

failures (WER > 0.5): 6L 2 ({'other': 2}), small 0 ({})

## hard set by tag (mean CER)

| tag | n | 6L | small |
|---|---|---|---|
| g_j | 60 | 0.0038 | 0.0021 |
| long | 38 | 0.0029 | 0.0048 |
| names | 60 | 0.0058 | 0.0083 |
| numbers | 60 | 0.0205 | 0.0318 |
| palatal | 60 | 0.0180 | 0.0216 |
| question | 60 | 0.0481 | 0.0036 |

## worst regressions (by CER increase)

- Unai [hard61] CER 0.00→0.14: Autobusa hamarrak eta laurden aterako da Bergaran. | small: Aukudu samarrak eta laurden aterako da bere garan.
- Maider [hard151] CER 0.00→0.14: Autobusa hamabiak eta laurden aterako da Gasteizen. | small: Aupuru Zamabiak eta laurden aterako da Gasteizen.
- Unai [hard45] CER 0.00→0.12: Autobusa zortziak eta laurden aterako da Durangon. | small: Aukugu zortziak eta laurden aterako da Durangon.
- bank0027 [hard158] CER 0.00→0.12: Autobusa zeroak laurden gutxi aterako da Bergaran. | small: Aupuru ezeroak laurden gutxi aterako da bergaran.
- bank0070 [hard45] CER 0.00→0.12: Autobusa zortziak eta laurden aterako da Durangon. | small: Aupuru zortziak eta laurden aterako da Durangon.
- Unai [main332] CER 0.00→0.10: Hezkuntza, dohain eta laiko bihurtuz. | small: Huntza, dohain eta laiko bihurtuz.
- eum_08886 [main124] CER 0.00→0.10: Gizarte aldaketa hori bati-bat gizonaren alorrean antzeman zen, izan e | small: Gizarte aldaketa hori bati bat gizonaren alorrean antzeman zen, izan e
- Aitor [hard158] CER 0.00→0.09: Autobusa zeroak laurden gutxi aterako da Bergaran. | small: Aupu busteroak laurden gutxi aterako da bergaran.
- Unai [hard93] CER 0.00→0.09: Autobusa ordu bata eta erdiak aterako da Gasteizen. | small: Aukurusordu bata eta erdiak aterako da Gasteizen.
- bank0027 [hard143] CER 0.03→0.12: Azken urteetan egindako ikerketek Schillerrek estetikaren inguruan egi | small: Azken urteetan egindako ikerketek, sillerrek, estilerrek, estetekaren 
- bank0083 [hard144] CER 0.00→0.08: Autobusa hemezortziak laurden gutxi aterako da Baionan. | small: Au Pubes hemezortziak laurden gutxi aterako da Baionan.
- Unai [main372] CER 0.01→0.09: Europako leku askotan, Erdi Aroan haurren bat egun batez abade ez ezik | small: Europako leku askotan, Erdi Aroan haurren batez abade ez ezik, aita za
