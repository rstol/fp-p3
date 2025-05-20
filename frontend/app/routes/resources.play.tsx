import invariant from 'tiny-invariant';

import type { ActionFunctionArgs } from 'react-router';
import { BASE_URL, PlayActions } from '~/lib/const';

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method.toLowerCase() === 'post', 'Invalid request method - expected POST');

  const formData = await request.formData();

  const { _action, ...values } = Object.fromEntries(formData);
  invariant(_action && typeof _action === 'string', 'Missing or invalid action in payload');
  invariant(
    values.eventId && typeof values.eventId === 'string',
    'Missing or invalid eventId in payload',
  );
  invariant(
    values.gameId && typeof values.gameId === 'string',
    'Missing or invalid gameId in payload',
  );
  console.log(values);
  switch (_action) {
    case PlayActions.UpdateAllPlayFields: {
      let { note, clusters, gameId, eventId } = values;
      clusters = JSON.parse(clusters as string);
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
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    }
    case PlayActions.UpdatePlayNote: {
      const { note, gameId, eventId } = values;

      const payload = {
        note: note,
      };
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    }
    case PlayActions.UpdatePlayCluster:
      let { clusters, gameId, eventId } = values;
      clusters = JSON.parse(clusters as string);
      invariant(
        Array.isArray(clusters) && clusters.length > 0,
        'Missing or invalid clusters in payload',
      );

      // Set cluster id to null for new cluster
      const cluster = clusters[0];
      const payload = {
        cluster_id: cluster.id.startsWith('new_cluster') ? null : cluster.id,
        cluster_name: cluster.text,
      };
      const backendResponse = await fetch(`${BASE_URL}/plays/${gameId}/${eventId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      return backendResponse; // No actual data just 200 or 40x
    default:
      break;
  }
  return;
}
