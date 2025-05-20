import invariant from 'tiny-invariant';

import type { ActionFunctionArgs } from 'react-router';
import { BASE_URL } from '~/lib/const';

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method.toLowerCase() === 'post', 'Invalid request method - expected POST');

  console.log(request);

  const formData = await request.formData();

  const data = Object.fromEntries(formData);

  console.log('data', data);
  const { clusterId, clusterLabel } = data;

  invariant(clusterId && typeof clusterId === 'string', 'Missing or invalid clusterId in payload');
  invariant(
    clusterLabel && typeof clusterLabel === 'string',
    'Missing or invalid clusterLabel in payload',
  );

  // Set cluster id to null for new cluster
  const payload = {
    cluster_label: clusterLabel,
  };
  const backendResponse = await fetch(`${BASE_URL}/cluster/${clusterId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return backendResponse; // No actual data just 200 or 40x
}
