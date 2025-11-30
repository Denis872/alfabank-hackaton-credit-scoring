"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/app/components/ui/dialog";

import { DialogProps } from "@radix-ui/react-dialog";
import { Button } from "../ui";
import { scripts } from "./data";
import { CardType, ScriptType } from "./types";

interface ModalProps extends DialogProps {
  scriptId: number;
  offerId: number;
}

export function ScriptModal({
  open,
  onOpenChange,
  scriptId,
  offerId,
}: ModalProps) {
  const text = scripts[offerId as CardType][scriptId as ScriptType];

  return (
    <Dialog open={open}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Скрипт продажи</DialogTitle>
        </DialogHeader>
        <div style={{ whiteSpace: "pre-wrap" }}>{text}</div>
        <Button
          className="h-[52px] w-[200px] bg-black text-white font-semibold hover:bg-gray-800"
          onClick={() => onOpenChange?.(false)}
        >
          Закрыть
        </Button>
      </DialogContent>
    </Dialog>
  );
}
