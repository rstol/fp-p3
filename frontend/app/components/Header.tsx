import type { ReactElement } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '~/components/ui/tabs';

const getTabValue = (value: string) => value.toLowerCase().replace(' ', '-');

export default function Header({
  tabs,
  defaultTab,
}: {
  tabs: { title: string; children: ReactElement }[];
  defaultTab?: string;
}) {
  defaultTab = defaultTab ?? tabs[0].title;
  console.log(defaultTab);
  return (
    <div className="space-y-4">
      {/* <h1 className="text-center font-medium">Basketball Play Analyzer</h1> */}
      <Tabs defaultValue={getTabValue(defaultTab)} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          {tabs.map((tab) => (
            <TabsTrigger key={getTabValue(tab.title)} value={getTabValue(tab.title)}>
              {tab.title}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabs.map((tab) => (
          <TabsContent value={getTabValue(tab.title)} key={getTabValue(tab.title)}>
            {tab.children}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
