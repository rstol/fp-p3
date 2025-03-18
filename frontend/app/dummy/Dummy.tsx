import { useState, useEffect } from 'react';
import DataChoiceComponent from '~/components/DataChoice';
import ScatterPlot from '~/components/ScatterPlot';
import { BASE_URL } from '~/lib/const';
import type { DataArray } from '~/types/data';

interface FetchError {
  message: string;
  status?: number;
}

async function postPoints(
  choice: string
): Promise<{ data?: DataArray; error?: FetchError }> {
  try {
    console.log(`${BASE_URL}/data/${choice}`);
    const response = await fetch(`${BASE_URL}/data/${choice}`);
    if (!response.ok) {
      return {
        error: {
          message: `Failed to fetch data: ${response.statusText}`,
          status: response.status,
        },
      };
    }
    const data = await response.json();
    return { data };
  } catch (err) {
    return {
      error: {
        message:
          err instanceof Error
            ? err.message
            : 'An unexpected error occurred',
      },
    };
  }
}

export function Dummy() {
  const [exampleData, setExampleData] = useState<
    DataArray | undefined
  >();
  const [dataChoice, setDataChoice] = useState<string>();
  const [error, setError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function fetchData() {
      if (!dataChoice) return;

      setIsLoading(true);
      setError(undefined);

      const result = await postPoints(dataChoice);

      if (result.error) {
        setError(result.error.message);
        setExampleData(undefined);
      } else if (result.data) {
        setExampleData(result.data);
      }

      setIsLoading(false);
    }

    fetchData();
  }, [dataChoice]);

  function choiceMade(choice: string) {
    setDataChoice(choice);
  }

  return (
    <div className="w-screen h-screen flex flex-col items-center bg-[#2692db] font-['Neucha']">
      <header className="uppercase text-4xl mt-24 mb-8 text-white z-10 text-center lg:text-3xl lg:my-4">
        K-Means clustering
      </header>
      <DataChoiceComponent onChoiceMade={choiceMade} />

      {isLoading && (
        <div className="mt-4 text-white text-lg">Loading data...</div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-500 text-white rounded-lg shadow-lg">
          {error}
        </div>
      )}

      {!isLoading && !error && (
        <ScatterPlot width={1100} height={550} data={exampleData} />
      )}
    </div>
  );
}
