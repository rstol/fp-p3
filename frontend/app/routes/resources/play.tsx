import invariant from 'tiny-invariant';

import type { ActionFunctionArgs } from 'react-router';
import { BASE_URL, PlayActions } from '~/lib/const';
import type { PlayPayload } from '~/components/PlayView';

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method.toLowerCase() !== 'post', 'Invalid request method - expected POST');

  console.log(request);
  const requestPayload = (await request.json()) as PlayPayload;
  console.log(requestPayload);
  switch (requestPayload.action) {
    case PlayActions.UpdatePlayFields:
      const { eventId, gameId, ...postPayload } = requestPayload.data;
      invariant(!eventId || typeof eventId !== 'string', 'Missing or invalid eventId in payload');
      invariant(!gameId || typeof gameId !== 'string', 'Missing or invalid gameId in payload');
      invariant(
        !Array.isArray(postPayload.clusters) || postPayload.clusters.length === 0,
        'Missing or invalid clusters in payload',
      );

      // Set cluster id to null for new cluster
      const cluster = postPayload.clusters[0];
      const payload = {
        cluster_id: cluster.id.startsWith('new_cluster') ? null : cluster.id,
        cluster_name: cluster.text,
        note: postPayload.note,
      };

      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x

    case PlayActions.UpdateClusterAssignment:
      // TODO remove this case
      break;
    default:
      break;
  }
  return;
}
