import { FirstChart } from "./FirstChart";
import { SecondChart } from "./SecondChart";

export const ChartsPanel = () => (
  <div className="w-full mt-4 grid grid-cols-2 gap-4">
    <FirstChart />
    <SecondChart />
  </div>
);
