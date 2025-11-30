import {
  appleBurnScript,
  appleHoldScript,
  appleSellScript,
  orangeBurnScript,
  orangeHoldScript,
  orangeSellScript,
  stickerBurnScript,
  stickerHoldScript,
  stickerSellScript,
  travelBurnScript,
  travelHoldScript,
  travelSellScript,
} from "@/app/data";
import { CardType, ScriptType } from "./types";

export const scripts: Record<CardType, Record<ScriptType, string>> = {
  1: {
    1: travelSellScript,
    2: travelHoldScript,
    3: travelBurnScript,
  },
  2: {
    1: appleSellScript,
    2: appleHoldScript,
    3: appleBurnScript,
  },
  3: {
    1: orangeSellScript,
    2: orangeHoldScript,
    3: orangeBurnScript,
  },
  4: {
    1: stickerSellScript,
    2: stickerHoldScript,
    3: stickerBurnScript,
  },
};
