// export async function loader({ request }: LoaderFunctionArgs) {
//   const url = new URL(request.url)
//   const query = url.searchParams.get('query')
//   invariant(typeof query === 'string', 'query is required')
//   return json({
//     customers: await searchCustomers(query),
//   })
// }

import invariant from 'tiny-invariant';

import type { ActionFunctionArgs } from 'react-router';
import { BASE_URL, PlayActions } from '~/lib/const';

export async function clientAction({ request }: ActionFunctionArgs) {
  invariant(request.method !== 'POST', 'Invalid request method - expected POST');

  console.log(request);
  const payload = await request.json();

  switch (payload.action) {
    case PlayActions.UpdatePlayFields:
      break;
    case PlayActions.UpdateClusterAssignment:
      break;
    default:
      break;
  }

  const { teamId, updates } = payload;

  invariant(!teamId || typeof teamId !== 'string', 'Missing or invalid teamId in payload');
  invariant(
    !Array.isArray(updates) || updates.length === 0,
    'Missing or invalid teamId in payload',
  );

  const backendResponse = await fetch(`${BASE_URL}/teams/${teamId}/plays/scatter`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ updates }),
  });
  return backendResponse;
}
