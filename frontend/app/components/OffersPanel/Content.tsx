import { FromValues } from "@/app/types";
import { Controller, useFormContext } from "react-hook-form";
import { OfferItem } from "./OfferItem";

const offers = [
  {
    id: 1,
    title: "Alfa Travel",
    subtitle: "+200K / год",
    tags: ["Кэшбэк 7%", "Бесплатная"],
    img: "/credit_01",
  },
  {
    id: 2,
    title: "Карта Альфа и ЗЯ",
    subtitle: "+200K / год",
    tags: ["Кэшбэк 10%", "99₽"],
    img: "/credit_02",
  },
  {
    id: 3,
    title: "Апельсиновая карта",
    subtitle: "+200K / год",
    tags: ["Кэшбэк 10%", "99₽"],
    img: "/credit_03",
  },
  {
    id: 4,
    title: "Альфа-Стикер",
    subtitle: "+200K / год",
    tags: ["Бесплатная"],
    img: "/credit_04",
  },
];

export const Content = () => {
  const { control } = useFormContext<FromValues>();

  return (
    <div className="grid grid-cols-1 gap-3">
      {offers.map((offer) => (
        <Controller
          name="offerId"
          control={control}
          render={({ field: { value, onChange } }) => (
            <OfferItem
              onChange={() => onChange(offer.id)}
              selected={offer.id === value}
              key={offer.id}
              {...offer}
            />
          )}
          key={offer.id}
        />
      ))}
    </div>
  );
};
