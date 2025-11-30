import { MessageCirclePlus, PhoneOutgoing } from "lucide-react";
import { Avatar, Button } from "../ui";

const getRandUserImage = () => {
  const randIdx = Math.floor(Math.random() * 4) + 1;

  return `/users/user_0${randIdx}.png`;
};

export const Header = () => (
  <div className="w-full flex flex-row justify-between items-center pb-7 border-b border-b-gray-200">
    <div className="flex gap-4 items-center">
      <Avatar src={getRandUserImage()} />
      <div>
        <div className="text-lg font-semibold">
          Третьяков Артём Александрович
        </div>
        <div className="text-sm text-gray-400">
          Дебетовая Alfa • 6400 0540 3391 6772
        </div>
      </div>
    </div>
    <div className="flex items-center gap-3">
      <Button>
        <PhoneOutgoing className="size-4" />
        Звонок
      </Button>
      <Button>
        <MessageCirclePlus className="size-4" />
        Чат
      </Button>
    </div>
  </div>
);
