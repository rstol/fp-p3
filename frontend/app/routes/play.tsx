import invariant from 'tiny-invariant';

import type { ActionFunctionArgs } from 'react-router';
import { BASE_URL, PlayActions } from '~/lib/const';
import type { PlayPayload } from '~/components/PlayView';

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method.toLowerCase() === 'post', 'Invalid request method - expected POST');

  console.log(request);
  // const requestPayload = (await request.json()) as PlayPayload;

  const requestPayload = await request.formData();

  console.log(requestPayload);


  switch (requestPayload.get('action')) {
    case PlayActions.UpdatePlayFields:

      const data = JSON.parse(requestPayload.get('data') as string);
      console.log("data", data)
      const eventId = data.eventId;
      const gameId = data.gameId;
      const clusters = data.clusters;
      const note = data.note;

      invariant(eventId && typeof eventId === 'number', 'Missing or invalid eventId in payload');
      invariant(gameId && typeof gameId === 'number', 'Missing or invalid gameId in payload');
      invariant(
        Array.isArray(clusters) && clusters.length > 0,
        'Missing or invalid clusters in payload',
      );

      // Set cluster id to null for new cluster
      const cluster = clusters[0];
      const payload = {
        cluster_id: cluster.id.startsWith('new_cluster') ? null : cluster.id,
        cluster_name: cluster.text,
        note: note,
      };

      console.log(`${BASE_URL}/plays/${gameId}/${eventId}`)
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      console.log("getttttttttttttting there")
      return backendResponse; // No actual data just 200 or 40x

    case PlayActions.UpdateClusterAssignment:
      // TODO remove this case
      break;
    default:
      break;
  }
  return;
}
